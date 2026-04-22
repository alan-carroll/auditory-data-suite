# Disable Kivy's argument parser so the --db / --analysis-id / --ic
# flags reach argparse in __main__ instead of being consumed (and
# rejected) by Kivy at import time.
import os
os.environ["KIVY_NO_ARGS"] = "1"

# Reduce Kivy's logging spam to the terminal
os.environ.setdefault("KIVY_LOG_MODE", "PYTHON")
os.environ.setdefault("KIVY_NO_CONSOLELOG", "1")
os.environ.setdefault("KIVY_NO_FILELOG", "1")

# Non-interactive MPL backend
os.environ.setdefault("MPLBACKEND", "Agg")

import logging
import itertools
from collections import namedtuple

from logging_utils import configure_file_logging, install_excepthooks
from compat import install_distutils_version_shim
import numpy as np
from matplotlib import colors as mpl_colors, colormaps as mpl_colormaps
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
import cmocean
from db_adapter import JSONStore
from site_model import SiteModel, StimConfig

# kivy_garden.matplotlib still imports distutils.version on Python 3.12+.
install_distutils_version_shim()
from kivy_garden.matplotlib.backend_kivyagg import (
    FigureCanvasKivyAgg as FigureCanvas,
)

configure_file_logging("map_gui_log.log", level=logging.ERROR)
install_excepthooks()
logger = logging.getLogger(__name__)

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics import Color, Line, Mesh
from kivy.utils import get_color_from_hex as hex2rgb
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.slider import Slider

logging.getLogger("kivy").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

if Window is not None:
    Window.clearcolor = (1, 1, 1, 1)
    Window.size = (max(Window.width, 1500), max(Window.height, 950))


LineTuple = namedtuple("LineTuple",
                       ["line", "color", "x_norm", "y_norm", "site_number"])
MeshTuple = namedtuple("MeshTuple",
                       ["mesh", "color", "x_norm", "y_norm", "site_number"])


class FieldSelectionApp(App):
    def __init__(self, db_path, analysis_id, is_ic):
        """
        Uses `ScreenManager` to allow swapping between main Map GUI and
        Site-specific GUIs.
        """
        super(FieldSelectionApp, self).__init__()
        self.SM = ScreenManager()
        self.map_screen = MapScreen(db_path, analysis_id, is_ic,
                                    name="Map")
        self.SM.add_widget(self.map_screen)
        self.SM.current = "Map"

    def build(self):
        return self.SM


class MapScreen(Screen):
    def __init__(self, db_path, analysis_id, is_ic, **kwargs):
        """Container for main application."""
        super(MapScreen, self).__init__(**kwargs)
        self.add_widget(FieldSelectionGUI(db_path, analysis_id, is_ic))


class SiteScreen(Screen):
    def __init__(self, gui_instance, num, densetc_plot, **kwargs):
        """Site-specific view permitting analysis updates."""
        super(SiteScreen, self).__init__(**kwargs)

        self.unsaved_changes = False
        self.map_number = num
        self.gui_instance = gui_instance

        # Arrange GUI
        self.layout = BoxLayout(orientation="vertical")
        self.top_menu_layout = BoxLayout(orientation="horizontal", 
                                         size_hint=(1, 0.08))
        self.back_button = Button(text="Back to Map", size_hint=(0.15, 1))
        self.mark_site_toggle = ToggleButton(text="Mark Site", 
                                             size_hint=(0.13, 1))
        self.save_changes_button = Button(text="Save Changes", 
                                          size_hint=(0.13, 1), 
                                          background_normal="",
                                          background_color=[0.1, 0.4, 0.1, 1], 
                                          disabled=True)
        self.auto_tc_button = Button(text="Auto-analyze TC",
                                     size_hint=(0.13, 1))
        self.reset_button = Button(text="Reset", size_hint=(0.13, 1), 
                                   background_normal="", 
                                   background_color=[0.25, 0.05, 0.1, 1], 
                                   disabled=True)
        self.back_button.bind(on_release=self.change_screen)
        self.mark_site_toggle.bind(on_release=self.on_mark_toggle)
        self.save_changes_button.bind(on_release=self.save_changes)
        self.auto_tc_button.bind(on_release=self.auto_tc_analyze)
        self.reset_button.bind(on_release=self.on_reset)

        self.top_menu_layout.add_widget(self.back_button)
        self.top_menu_layout.add_widget(self.save_changes_button)
        self.top_menu_layout.add_widget(self.mark_site_toggle)
        self.top_menu_layout.add_widget(self.auto_tc_button)
        self.top_menu_layout.add_widget(self.reset_button)
        self.layout.add_widget(self.top_menu_layout)

        self.tools_layout = BoxLayout(orientation="vertical", 
                                      size_hint=(1, 0.11))

        self.site_label = Label(text=f"Site {self.map_number}", 
                                color=[0, 0, 0, 1], size_hint=(0.15, 1))

        tool_button_layout = BoxLayout(orientation="horizontal", 
                                       size_hint=(0.5, 1))
        self.pick_cf_button = Button(text="Pick new CF")
        self.pick_cf_button.bind(on_release=self.pick_cf)
        tool_button_layout.add_widget(self.pick_cf_button)

        contour_layout = BoxLayout(orientation="vertical", size_hint=(0.1, 1))
        self.contour_checkbox = CheckBox(active=False, size_hint=(1, 0.5))
        self.contour_checkbox.bind(active=self._on_plot_flag_checkbox)
        self.contour_checkbox_label = Label(text="TC Contour: ", 
                                            color=[0, 0, 0, 1], 
                                            size_hint=(1, 0.5))
        contour_layout.add_widget(self.contour_checkbox_label)
        contour_layout.add_widget(self.contour_checkbox)

        smooth_tc_layout = BoxLayout(orientation="vertical", 
                                     size_hint=(0.1, 1))
        self.smooth_tc_checkbox = CheckBox(active=False, size_hint=(1, 0.5))
        self.smooth_tc_checkbox.bind(active=self._on_plot_flag_checkbox)
        self.smooth_tc_checkbox_label = Label(text="Smooth TC: ", 
                                              color=[0, 0, 0, 1], 
                                              size_hint=(1, 0.5))
        smooth_tc_layout.add_widget(self.smooth_tc_checkbox_label)
        smooth_tc_layout.add_widget(self.smooth_tc_checkbox)

        lineplot_layout = BoxLayout(orientation="vertical", size_hint=(0.1, 1))
        self.lineplot_checkbox = CheckBox(active=False, size_hint=(1, 0.5))
        self.lineplot_checkbox.bind(active=self._on_plot_flag_checkbox)
        self.lineplot_checkbox_label = Label(text="TC Line Plot: ", 
                                             color=[0, 0, 0, 1], 
                                             size_hint=(1, 0.5))
        lineplot_layout.add_widget(self.lineplot_checkbox_label)
        lineplot_layout.add_widget(self.lineplot_checkbox)

        heatmap_layout = BoxLayout(orientation="vertical", size_hint=(0.1, 1))
        self.heatmap_checkbox = CheckBox(active=False, size_hint=(1, 0.5))
        self.heatmap_checkbox.bind(active=self._on_plot_flag_checkbox)
        self.heatmap_checkbox_label = Label(text="TC Heatmap: ", 
                                            color=[0, 0, 0, 1], 
                                            size_hint=(1, 0.5))
        heatmap_layout.add_widget(self.heatmap_checkbox_label)
        heatmap_layout.add_widget(self.heatmap_checkbox)

        bw_layout = BoxLayout(orientation="vertical", size_hint=(0.1, 1))
        self.bw_checkbox = CheckBox(active=True, size_hint=(1, 0.5))
        self.bw_checkbox.bind(active=self._on_plot_flag_checkbox)
        self.bw_checkbox_label = Label(text="Show BWs: ", color=[0, 0, 0, 1], 
                                       size_hint=(1, 0.5))
        bw_layout.add_widget(self.bw_checkbox_label)
        bw_layout.add_widget(self.bw_checkbox)

        # checkbox -> SitePlot flag it controls. Populated here (after
        # all five checkboxes exist) rather than at each bind site.
        self._plot_flag_checkboxes = {
            self.contour_checkbox:   "use_contour",
            self.smooth_tc_checkbox: "use_smooth_tc",
            self.lineplot_checkbox:  "use_lineplot",
            self.heatmap_checkbox:   "use_heatmap",
            self.bw_checkbox:        "use_bw",
        }

        bin_size_layout = BoxLayout(orientation="horizontal", 
                                    size_hint=(0.1, 1))
        self.bin_size_label = Label(text="Bin Size", color=[0, 0, 0, 1], 
                                    size_hint=(0.7, 1))
        self.bin_size_spinner = Spinner(text="1 ms", values=("1 ms", "5 ms"), 
                                        size_hint=(0.3, 1))
        self.bin_size_spinner.bind(text=self.change_bin_size)
        bin_size_layout.add_widget(self.bin_size_label)
        bin_size_layout.add_widget(self.bin_size_spinner)

        bubble_slider_layout = BoxLayout(orientation="horizontal")
        self.bubble_slider = Slider(size_hint=(0.975, 1), min=1, max=100, 
                                    value=40, step=1)
        self.bubble_slider.bind(value=self.change_bubble_size)
        self.bubble_slider_label = Label(text="Bubble Size", 
                                         color=[0, 0, 0, 1], 
                                         size_hint=(None, 1))
        bubble_slider_layout.add_widget(self.bubble_slider_label)
        bubble_slider_layout.add_widget(self.bubble_slider)

        options_layout = BoxLayout(orientation="horizontal")
        options_layout.add_widget(self.site_label)
        options_layout.add_widget(tool_button_layout)
        options_layout.add_widget(contour_layout)
        options_layout.add_widget(smooth_tc_layout)
        options_layout.add_widget(lineplot_layout)
        options_layout.add_widget(heatmap_layout)
        options_layout.add_widget(bw_layout)
        options_layout.add_widget(bin_size_layout)

        self.tools_layout.add_widget(bubble_slider_layout)
        self.tools_layout.add_widget(options_layout)

        densetc_plot.max_bubble_size = 30
        self.densetc_plot = densetc_plot
        if self.densetc_plot.model.working.marked:
            self.mark_site_toggle.state = "down"
        self.layout.add_widget(self.densetc_plot)

        self.layout.add_widget(self.tools_layout)
        self.add_widget(self.layout)

        # Listen for user changes to analysis
        self.densetc_plot.on_changes_signal.connect(self.changes_made)
        self.densetc_plot.on_cf_pick_signal.connect(self.cf_picked)
        self._syncing_plot_flag_checkbox = False
        
    def redraw(self):
        """Re-draw plots."""
        self.densetc_plot.re_plot(axis_visible="on")
        self.densetc_plot.draw_canvas()

    def _on_plot_flag_checkbox(self, checkbox, checked):
        """
        Shared handler for the five TC-display checkboxes. Each flips
        one boolean on the detail SitePlot and redraws; which boolean
        is looked up from the checkbox instance.
        """
        if self._syncing_plot_flag_checkbox:
            return

        attr = self._plot_flag_checkboxes[checkbox]

        if attr == "use_lineplot":
            self.densetc_plot.use_lineplot = checked
            if checked:
                self.densetc_plot.use_heatmap = False
                self._syncing_plot_flag_checkbox = True
                self.heatmap_checkbox.active = False
                self._syncing_plot_flag_checkbox = False
        elif attr == "use_heatmap":
            self.densetc_plot.use_heatmap = checked
            if checked:
                self.densetc_plot.use_lineplot = False
                self._syncing_plot_flag_checkbox = True
                self.lineplot_checkbox.active = False
                self._syncing_plot_flag_checkbox = False
        else:
            setattr(self.densetc_plot, attr, checked)

        self.bubble_slider.disabled = (
            self.densetc_plot.use_lineplot or self.densetc_plot.use_heatmap
        )
        self.redraw()

    def on_mark_toggle(self, _event):
        """Event monitoring if site is 'marked' or not."""
        model = self.densetc_plot.model
        new_marked = (self.mark_site_toggle.state == "down")
        if new_marked != model.saved.marked:
            self.densetc_plot.on_changes_signal.send()
        model.working.marked = new_marked

    def on_reset(self, _event):
        """Event to reset any un-saved analysis changes made to a site."""
        self.unsaved_changes = False
        self.save_changes_button.disabled = True
        self.save_changes_button.background_color = [0.2, 0.65, 0, 1]
        self.reset_button.disabled = True
        self.reset_button.background_color = [0.25, 0.05, 0.1, 1]

        self.densetc_plot.model.reset()
        self.redraw()

        # Reset Marked toggle, if necessary
        self.mark_site_toggle.state = (
            "down" if self.densetc_plot.model.saved.marked else "normal")

    def change_bin_size(self, _spinner, value):
        """Show PSTH with 1 or 5 ms bin size."""
        self.densetc_plot.bin_size = 5 if value == "5 ms" else 1
        self.redraw()

    def pick_cf(self, _event):
        """Pick a new CF for the TC by clicking in plot."""
        Window.clearcolor = hex2rgb("#d1ffbd")
        self.densetc_plot.picking_cf = True
        self.pick_cf_button.disabled = True
        self.back_button.disabled = True

    def cf_picked(self, *_args, **_kwargs):
        """Update with new CF."""
        Window.clearcolor = (1, 1, 1, 1)
        self.pick_cf_button.disabled = False
        self.back_button.disabled = False
        self.redraw()

    def save_changes(self, *_args, **_kwargs):
        """
        Persist working state to the DB and commit it as the new
        reset baseline.
        """
        import datetime
        import analysis_functions as afunc
        model = self.densetc_plot.model
        st = model.working
        cfg = model.config
        freqs = np.asarray(cfg.frequencies_hz)
        ints = np.asarray(cfg.intensities_db)

        self.unsaved_changes = False
        self.save_changes_button.disabled = True
        self.save_changes_button.background_color = [0.2, 0.65, 0, 1]
        self.reset_button.disabled = True
        self.reset_button.background_color = [0.25, 0.05, 0.1, 1]

        if not st.continuous_bw_idx or st.continuous_bw_idx[0] is None:
            # Manual edits were made without re-running auto-analyze,
            # so continuous BW was never refreshed.
            st.continuous_bw_idx = afunc.ttest_analyze_tuning_curve(
                model.ttest_tc()).continuous_bw

        # Convert indices to physical units
        bw_khz, bw_oct = afunc.bw_idx_to_units(st.bw_idx, freqs)

        try:
            cont_bw_khz = [(freqs[b] / 1000).tolist()
                           for b in st.continuous_bw_idx]
            cont_bw_oct = [afunc.get_bandwidth(*freqs[b]).tolist()
                           for b in st.continuous_bw_idx]
        except TypeError:
            # Auto-analysis found no regions; persist as absent.
            st.continuous_bw_idx = [None, None]
            cont_bw_khz = [None, None]
            cont_bw_oct = None
        
        # Guard cf/thresh against None
        cf_khz = (None if st.cf_idx is None 
                  else freqs[st.cf_idx] / 1000)
        threshold_db = (None if st.thresh_idx is None 
                        else ints[st.thresh_idx].tolist())
        update_doc = {
            "cf_khz": cf_khz,
            "threshold_db": threshold_db,
            "cf_idx": st.cf_idx,
            "threshold_idx": st.thresh_idx,
            "continuous_bw_khz": cont_bw_khz,
            "continuous_bw_idx": st.continuous_bw_idx,
            "continuous_bw_octave": cont_bw_oct,
            "onset_ms": st.onset,
            "peak_ms": st.peak,
            "offset_ms": st.offset,
            "peak_driven_rate_hz": st.peak_driven_rate,
            "marked": st.marked,
        }
        for lvl in cfg.bw_levels_db:
            update_doc[f"bw{lvl}_idx"] = st.bw_idx[lvl]
            update_doc[f"bw{lvl}_khz"] = bw_khz[lvl]
            update_doc[f"bw{lvl}_octave"] = bw_oct[lvl]

        self.gui_instance.densetc_analysis_collection.update_one(
            {"analysis_id": self.gui_instance.analysis_id,
             "number": self.map_number},
            {"$set": update_doc}
        )

        self.gui_instance.analysis_metadata_collection.update_one(
            {"_id": self.gui_instance.analysis_id},
            {"$set": {"last_modified": str(datetime.datetime.now())}}
        )

        model.commit()
        self.redraw()

    def auto_tc_analyze(self, *_args, **_kwargs):
        """
        Re-run the TC auto-analysis at the current latency window and
        load results into working state. Nothing persists until Save.
        """
        import analysis_functions as afunc
        model = self.densetc_plot.model
        st = model.working
        cfg = model.config
        self.densetc_plot.on_changes_signal.send()
        r = afunc.ttest_analyze_tuning_curve(model.ttest_tc())
        st.cf_idx = r.cf
        st.thresh_idx = r.thresh
        st.bw_idx = r.bw_idx
        st.continuous_bw_idx = r.continuous_bw
        self.redraw()

    def changes_made(self, *_args, **_kwargs):
        """Event signaling analysis changes have been made."""
        self.unsaved_changes = True
        self.save_changes_button.disabled = False
        self.save_changes_button.background_color = [0.2, 0.65, 0, 1]
        self.reset_button.disabled = False
        self.reset_button.background_color = [0.7, 0.1, 0.15, 1]

    def change_bubble_size(self, _slider, value):
        """Event signaling update to max bubble size for TC plot."""
        self.densetc_plot.max_bubble_size = value
        self.densetc_plot.update_bubble_size()
        self.densetc_plot.figure_canvas.draw()

    def change_screen(self, _event):
        """
        Close Site-specific screen and return to Map GUI overview.
        Updates persist between views.
        """
        overview = self.gui_instance.plot_dict[self.map_number]
        overview.re_plot()
        try:
            overview.figure_canvas.draw()
        except ValueError:
            # Non-responsive sites occasionally raise on draw; harmless.
            pass

        self.gui_instance.flash_signal.send(
            self.map_number,
            unsaved_changes=self.unsaved_changes,
            marked=self.densetc_plot.model.working.marked)
        self.densetc_plot.active = False
        self.manager.switch_to(self.gui_instance.parent)

    def on_pre_enter(self, *args):
        """Ready Site plots prior to switching GUI screens."""
        self.densetc_plot.active = True
        self.densetc_plot.ensure_rendered(axis_visible="on")


class FieldSelectionGUI(BoxLayout):
    def __init__(self, db_path, analysis_id, is_ic):
        """
        Main application showing all Sites for a Map.
        Permits Auditory Field selection and overview of map properties and 
        analysis.

        DB file, analysis, and cortical/IC choice are selected in the
        CLI (map_analysis._pick_map_and_analysis) before this process
        launches, and arrive here via argv. The GUI has no file or
        analysis picker of its own; switching maps means closing the
        window and picking again from the CLI.
        """
        super(FieldSelectionGUI, self).__init__(orientation="horizontal")
        # Lazily wired when the first detail screen is opened.
        self.flash_signal = None
        self.flash_lw = 0
        self.flash_times = 0
        self.flash_line_color = None
        self.flash_mesh_color = None
        self.flash_mesh_alpha = 1
        self.flash_line = None
        self.flash_mesh = None
        self.flash_clock_event = None

        self._db_path = db_path
        self.analysis_id = analysis_id
        self.ic_bool = is_ic    # Changes coloring of histograms based on latency
        self.map_loaded = False
        self.subject_database = None
        self.map_metadata_collection = None
        self.map_metadata = None
        self.sites_collection = None
        self.densetc_analysis_collection = None
        self.densetc_data_collection = None
        self.analysis_metadata_collection = None
        self.project_configuration = None
        self.sites = None
        self.densetc_data = None
        self.densetc_analysis = None

        self.counter = 0
        self.site_screens = {}
        self.site_models = {}
        self.stim_config = None
        self._site_render_iter = None
        self._render_batch_size = 4
        self._total_sites = 0
        self._rendered_sites = 0
        self.load_popup = None
        self.load_popup_label = None
        self._syncing_plot_flag_toggles = False

        # Start with marks_active. Can be set to False before loading a map by
        # hitting the Show Fields button
        self.marks_active = True

        self.plot_dict = {}
        self.vor_lines = {}
        self.vor_meshes = {}
        
        # Used to control whether a cell is interactive or not.
        self.vor_active = {}

        self.unsaved_line_color = "#f7022a"  # xkcd:cherry red
        self.unsaved_mesh_color = "#cfff04"  # xkcd:neon yellow

        self.fields = ("A1", "VAF", "PAF", "AAF", "SRAF", "NAR", "Other", 
                       "Mark")
        colors = [
            "#3e82fc",  # A1 : xkcd:dodger blue
            "#ffff81",  # VAF : xkcd:butter
            "#90fda9",  # PAF : xkcd:foam green
            "#fc86aa",  # AAF : xkcd:pinky
            "#edc8ff",  # SRAF : xkcd:light lilac
            "#5a7d9a",  # NAR : xkcd:steel blue
            "#b04e0f",  # Other : xkcd:burnt sienna
            "#c1fd95",  # Mark: xkcd:celery
        ]
        line_colors = [
            "#0348c9",  # A1
            "#ffff00",  # VAF
            "#37fb65",  # PAF
            "#fa3872",  # AAF
            "#c44dff",  # SRAF
            "#394e60",  # NAR
            "#5e2908",  # Other
            "#60dc04",  # Mark
        ]
        self.map_sets = {field: set() for field in self.fields}
        self.field_colors = {field: color for field, color in 
                             zip(self.fields, colors)}
        self.field_line_colors = {field: color for field, color in 
                                  zip(self.fields, line_colors)}

        # Arrange GUI
        tools = StackLayout(orientation="lr-tb", size_hint=(0.075, 1))
        self.tools_panel = tools
        self.cf_spinner_label = Label(text="CF\n Colormap", 
                                      color=[0, 0, 0, 1], 
                                      size_hint=(1, 0.06), 
                                      halign="center")
        self.cf_colormap_dropdown = Spinner(
            text="viridis",
            size_hint=(1, 0.06),
            values=("viridis", "jet", "plasma", "inferno", "magma", "bone",
                    "cool", "tab20", "cubehelix", "gist_ncar"))
        self.heatmap_spinner_label = Label(text="Heatmap\n Colormap", 
                                           color=[0, 0, 0, 1], 
                                           size_hint=(1, 0.06),
                                           halign="center")
        self.heatmap_colormap_dropdown = Spinner(
            text="inferno", 
            size_hint=(1, 0.06), 
            values=("inferno", "viridis", "plasma", "magma", "ocean", 
                    "gnuplot2", "cubehelix", "jet", "bone", "gray"))
        self.cf_colormap_dropdown.bind(text=self._on_colormap)
        self.heatmap_colormap_dropdown.bind(text=self._on_colormap)

        self.toggle = ToggleButton(text="Select", group="paint", 
                                   size_hint=(1, 0.12))
        self.deselect_toggle = ToggleButton(text="Deselect", group="paint", 
                                            size_hint=(1, 0.12))
        self.show_figure_toggle = ToggleButton(text="Show\nFigure", 
                                               group="paint", 
                                               size_hint=(1, 0.05), 
                                               halign="center")
        self.hide_figure_toggle = ToggleButton(text="Hide\nFigure", 
                                               group="paint", 
                                               size_hint=(1, 0.05), 
                                               halign="center")
        self.export_map_num_button = Button(text="Save Fields /\n Marks", 
                                            size_hint=(1, 0.06), 
                                            halign="center")
        self.export_map_num_button.bind(on_release=self.export_map)
        self.increase_figsize_button = Button(text="+ Fig", 
                                              size_hint=(0.5, 0.07))
        self.decrease_figsize_button = Button(text="- Fig", 
                                              size_hint=(0.5, 0.07))
        self.increase_figsize_button.bind(on_release=self.increase_figsize)
        self.decrease_figsize_button.bind(on_release=self.decrease_figsize)

        self.field_alpha_label = Label(text="Field Alpha", color=[0, 0, 0, 1], 
                                       size_hint=(1, 0.03))
        self.field_alpha_slider = Slider(size_hint=(1, 0.04), min=0, max=100, 
                                         value=100, step=5, value_track=True, 
                                         value_track_color=[1, 0, 0, 1])
        self.field_alpha_slider.bind(value=self.change_field_alpha)

        self.field_spinner_label = Label(text="Field\nSelection", 
                                         color=[0, 0, 0, 1], 
                                         size_hint=(1, 0.04),
                                         halign="center")
        self.field_spinner = Spinner(text="Mark", values=self.fields, 
                                     size_hint=(1, 0.06))
        self.field_spinner.bind(text=self.check_mark_or_field)

        tools.add_widget(self.cf_spinner_label)
        tools.add_widget(self.cf_colormap_dropdown)
        tools.add_widget(self.toggle)
        tools.add_widget(self.show_figure_toggle)
        tools.add_widget(self.decrease_figsize_button)
        tools.add_widget(self.increase_figsize_button)
        tools.add_widget(self.hide_figure_toggle)
        tools.add_widget(self.deselect_toggle)
        tools.add_widget(self.heatmap_spinner_label)
        tools.add_widget(self.heatmap_colormap_dropdown)
        tools.add_widget(self.export_map_num_button)
        tools.add_widget(self.field_alpha_label)
        tools.add_widget(self.field_alpha_slider)
        tools.add_widget(self.field_spinner_label)
        tools.add_widget(self.field_spinner)

        self.plot_tools_layout = StackLayout(orientation="lr-tb",
                                             size_hint=(0.075, 1))

        self.toggle_contour = ToggleButton(text="Contours", 
                                           size_hint=(1, 0.058))
        self.toggle_lineplot = ToggleButton(text="Line Plots", 
                                            size_hint=(1, 0.058))
        self.toggle_bw = ToggleButton(text="Bandwidths", size_hint=(1, 0.058), 
                                      state="down")
        self.toggle_smooth = ToggleButton(text="Smooth TC", 
                                          size_hint=(1, 0.058))
        self.toggle_heatmap = ToggleButton(text="Heatmap TC", 
                                           size_hint=(1, 0.058))
        # Each map-wide display toggle just flips one boolean on every
        # overview SitePlot and redraws.
        self._plot_flag_toggles = {
            self.toggle_contour:  "use_contour",
            self.toggle_lineplot: "use_lineplot",
            self.toggle_bw:       "use_bw",
            self.toggle_smooth:   "use_smooth_tc",
            self.toggle_heatmap:  "use_heatmap",
        }
        for tgl in self._plot_flag_toggles:
            tgl.bind(on_release=self._on_plot_flag_toggle)

        self.toggle_show_fields = ToggleButton(text="Show Fields", 
                                               group="fields_or_marks", 
                                               size_hint=(1, 0.12),
                                               allow_no_selection=False)
        self.toggle_show_marks = ToggleButton(text="Show Marks", 
                                              group="fields_or_marks", 
                                              size_hint=(1, 0.12),
                                              allow_no_selection=False)
        self.toggle_show_marks.state = "down"
        self.toggle_show_marks.bind(state=self.on_show_marks)
        self.toggle_show_fields.bind(state=self.on_show_fields)

        self.hide_fields_layout = BoxLayout(orientation="vertical", 
                                            size_hint=(1, 0.3))
        self.toggle_hide_dict = {}
        for field in self.fields:
            self.toggle_hide_dict[field] = ToggleButton(
                text=f"Hide {field}", size_hint=(1, 1/len(self.fields)))
            self.toggle_hide_dict[field].bind(on_release=self.on_hide_field)
            self.hide_fields_layout.add_widget(self.toggle_hide_dict[field])

        self.map_bubble_label = Label(text="Bubble Size", color=[0, 0, 0, 1], 
                                      size_hint=(1, 0.03))
        self.map_bubble_slider = Slider(size_hint=(1, 0.04), min=1, max=20, 
                                        value=6, step=2, value_track=True, 
                                        value_track_color=[1, 0, 0, 1])
        self.map_bubble_slider.bind(value=self.change_bubble_size)

        self.psth_y_label = Label(text="PSTH Min.\nY-Lim", color=[0, 0, 0, 1], 
                                  size_hint=(1, 0.04), halign="center")
        self.psth_y_spinner = Spinner(text="None", 
                                      values={"None", "10", "20", "30", "40"}, 
                                      size_hint=(1, 0.06))
        self.psth_y_spinner.bind(text=self.on_psth_ylim)
        self.plot_tools_layout.add_widget(self.toggle_contour)
        self.plot_tools_layout.add_widget(self.toggle_lineplot)
        self.plot_tools_layout.add_widget(self.toggle_bw)
        self.plot_tools_layout.add_widget(self.toggle_smooth)
        self.plot_tools_layout.add_widget(self.toggle_heatmap)
        self.plot_tools_layout.add_widget(self.toggle_show_fields)
        self.plot_tools_layout.add_widget(self.toggle_show_marks)
        self.plot_tools_layout.add_widget(self.hide_fields_layout)
        self.plot_tools_layout.add_widget(self.map_bubble_label)
        self.plot_tools_layout.add_widget(self.map_bubble_slider)
        self.plot_tools_layout.add_widget(self.psth_y_label)
        self.plot_tools_layout.add_widget(self.psth_y_spinner)

        self.map_canvas = MapLayout(gui=self, size_hint_x=None, size_hint_y=None)
        self.scroll = MapScroll(size_hint=(1, 1))
        self.scroll.add_widget(self.map_canvas)

        self.add_widget(tools)
        self.add_widget(self.scroll)
        self.add_widget(self.plot_tools_layout)
        # Defer until the window / GL context exist -- display_map()
        # builds hundreds of matplotlib figures and draws on the Kivy
        # canvas, which needs the event loop running.
        Clock.schedule_once(self._load)

    def _ensure_flash_signal(self):
        if self.flash_signal is None:
            import blinker
            self.flash_signal = blinker.signal("flash")
            self.flash_signal.connect(self.flash_cell)

    def _set_loading_state(self, active):
        self.tools_panel.disabled = active
        self.plot_tools_layout.disabled = active
        self.scroll.disabled = active

    def _show_load_popup(self):
        if self.load_popup is not None:
            return
        layout = BoxLayout(orientation="vertical", padding=18)
        self.load_popup_label = Label(
            text="",
            markup=True,
            halign="center",
            valign="middle",
            color=[0.96, 0.96, 0.96, 1],
            font_size="26sp",
        )
        self.load_popup_label.bind(
            size=self.load_popup_label.setter("text_size"))
        layout.add_widget(self.load_popup_label)
        self.load_popup = Popup(
            title="",
            content=layout,
            size_hint=(None, None),
            size=(360, 140),
            auto_dismiss=False,
            separator_height=0,
        )
        self.load_popup.open()

    def _hide_load_popup(self):
        if self.load_popup is None:
            return
        self.load_popup.dismiss()
        self.load_popup = None
        self.load_popup_label = None

    def _update_load_status(self):
        if self._total_sites and self.load_popup_label is not None:
            self.load_popup_label.text = (
                "[b]Loading overview[/b]\n"
                f"{self._rendered_sites}/{self._total_sites} sites"
            )

    def _center_scroll_view(self, _dt=0):
        if self.scroll.width <= 0 or self.scroll.height <= 0:
            return
        if self.map_canvas.width > self.scroll.width:
            self.scroll.scroll_x = 0.5
        else:
            self.scroll.scroll_x = 0
        if self.map_canvas.height > self.scroll.height:
            self.scroll.scroll_y = 0.5
        else:
            self.scroll.scroll_y = 1

    def _schedule_center_scroll_view(self):
        for delay in (0, 0.05, 0.2):
            Clock.schedule_once(self._center_scroll_view, delay)

    def _start_display_map_batches(self, _dt):
        self._show_load_popup()
        self._update_load_status()
        self._schedule_center_scroll_view()
        # Give Kivy a beat to paint the popup before the first matplotlib
        # batch starts chewing through the event loop.
        Clock.schedule_once(self._display_map_batch, 0.05)

    def _load(self, _dt):
        """
        Connect to the DB, read project config, fetch sites + data +
        analysis, and render. Runs once on the first frame via
        Clock.schedule_once so the Kivy window and GL context exist
        before display_map() starts building per-site figures and
        drawing voronoi meshes on the canvas.

        All user choices (DB file, cortical-vs-IC, which analysis)
        were made in the CLI before this process launched; see
        map_analysis._pick_map_and_analysis().
        """
        try:
            is_ic = self.ic_bool

            # --- DB connection & collection handles ------------------
            self.subject_database = JSONStore(self._db_path)
            self.map_metadata_collection = self.subject_database.metadata
            self.map_metadata = self.map_metadata_collection.get_only()
            self.analysis_metadata_collection = \
                self.subject_database.analysis_metadata

            if is_ic:
                # IC has no stored map dimensions or voronoi polygons,
                # so we make them up. 3000x1000 is just "tall and
                # narrow" to suit a depth column; doesn't persist back
                # to the database.
                self.map_metadata["map_height"] = 3000
                self.map_metadata["map_width"] = 1000
                self.densetc_data_collection = \
                    self.subject_database.densetc_IC_data
                self.densetc_analysis_collection = \
                    self.subject_database.densetc_IC_analysis
            else:
                self.sites_collection = self.subject_database.sites
                self.densetc_data_collection = \
                    self.subject_database.densetc_data
                self.densetc_analysis_collection = \
                    self.subject_database.densetc_analysis

            # --- Project configuration -------------------------------
            # Expect exactly one analysis metadata doc to carry a
            # configuration (the auto-analysis run).
            self.project_configuration = \
                self.analysis_metadata_collection.find_one(
                    {"configuration": {"$exists": True}})["configuration"]

            # --- Sites, data, analysis -------------------------------
            if is_ic:
                self.sites = self._build_ic_pseudo_sites()
            else:
                self.sites = list(self.sites_collection.find({}))

            # Keyed by site number so each SitePlot can look its own
            # data up without re-querying tinymongo per site.
            self.densetc_data = {
                d["number"]: d
                for d in self.densetc_data_collection.find({})}
            self.densetc_analysis = {
                a["number"]: a
                for a in self.densetc_analysis_collection.find(
                    {"analysis_id": self.analysis_id})}
            
            # One StimConfig shared by every SiteModel. Sweep length is
            # sniffed from a PSTH if the project config doesn't carry it.
            any_analysis = next(iter(self.densetc_analysis.values()))
            self.stim_config = StimConfig.from_project_config(
                self.project_configuration,
                fallback_sweep_ms=len(any_analysis["psth"]))

            self.display_map()

        except Exception as e:
            logger.exception("Failed to load map")
            InfoPopup(
                "Error",
                f"Failed to load map:\n{e}\n\n"
                "Close the GUI and retry from the CLI.").open()

    @property
    def paint_mode_active(self):
        """
        True when any of the stroke-painting toggles is engaged.
        """
        return any(t.state == "down" for t in (
            self.toggle,
            self.deselect_toggle,
            self.show_figure_toggle,
            self.hide_figure_toggle,
        ))

    def open_site_screen(self, site_number):
        """
        Switch from the Map overview to the detailed analysis view for a site.
        Built on the fly when first accessed.
        `self.parent` is the MapScreen that owns this layout.
        """
        if site_number not in self.site_screens:
            self._ensure_flash_signal()
            detail_plot = SitePlot(
                model=self.site_models[site_number],
                detailed_plot=True,
                is_ic=self.ic_bool,
                cf_cmap=self.cf_colormap_dropdown.text,
                heatmap_cmap=self.heatmap_colormap_dropdown.text,
                size_hint=(1, 1),
                pos_hint={"center_x": 0.5, "center_y": 0.5},
                height=1,
                width=2)
            self.site_screens[site_number] = SiteScreen(
                self, site_number, detail_plot,
                name=f"Site {site_number}")
        self.parent.manager.switch_to(self.site_screens[site_number])

    def _on_colormap(self, *_):
        """
        Shared handler for both colormap spinners. Reads both spinner
        values directly, so it doesn't matter which one fired the event.
        """
        cf_cmap = self.cf_colormap_dropdown.text
        heatmap_cmap = self.heatmap_colormap_dropdown.text
        for plot in self.plot_dict.values():
            plot.re_color(cf_cmap=cf_cmap, heatmap_cmap=heatmap_cmap)
            plot.figure_canvas.draw()
        for site in self.site_screens.values():
            # Detail plots aren't on screen; update state but skip redraw.
            site.densetc_plot.re_color(cf_cmap=cf_cmap,
                                       heatmap_cmap=heatmap_cmap)
            site.densetc_plot.mark_dirty()

    def check_mark_or_field(self, _spinner, value):
        """
        Quick function to check what user intends and help them out instead of 
        requiring unnecessary mouse-clicks.
        """
        if value == "Mark":
            # User wants to mark sites instead of select fields. 
            # Make sure marks are visible
            self.toggle_show_marks.state = "down"
            self.toggle_show_fields.state = "normal"
        else:  # User wants to select fields. Make sure fields are visible.
            self.toggle_show_fields.state = "down"
            self.toggle_show_marks.state = "normal"

    def on_show_fields(self, _toggle, state):
        """Event triggering visibility change of Auditory Field selections."""
        alpha_value = self.field_alpha_slider.value_normalized
        if state == "down":  # Show fields
            self.marks_active = False
            if self.field_spinner.text == "Mark":
                self.field_spinner.text = "A1"
            if self.sites is None:  # Map isn't loaded
                return

            for site in self.sites:
                site_number = site["number"]
                field_assigned = False
                for field in self.fields:
                    if field == "Mark":
                        continue
                    if site_number in self.map_sets[field]:
                        field_assigned = True
                        self.vor_meshes[site_number].color.rgb = \
                            hex2rgb(self.field_colors[field])
                        self.vor_meshes[site_number].color.a = alpha_value
                        self.vor_lines[site_number].line.width = 3
                        self.vor_lines[site_number].color.rgb = \
                            hex2rgb(self.field_line_colors[field])
                if not field_assigned:
                    # Leave site blank if no field has been assigned yet.
                    self.vor_meshes[site_number].color.rgb = [1, 1, 1]
                    self.vor_lines[site_number].line.width = 1.5
                    self.vor_lines[site_number].color.rgb = \
                        [0.435, 0.51, 0.541]  # xkcd:steel grey

        # Handle visibility of sites
        self.on_hide_field("field_selection")

    def on_show_marks(self, _toggle, state):
        """Event triggering visibility change of 'Marked' status for sites."""
        alpha_value = self.field_alpha_slider.value_normalized
        if state == "down":  # Show marks
            self.marks_active = True
            if self.field_spinner.text != "Mark":
                self.field_spinner.text = "Mark"
            if self.sites is None:  # Map isn't loaded
                return

            for site in self.sites:
                site_number = site["number"]
                if site_number in self.map_sets["Mark"]:
                    self.vor_meshes[site_number].color.rgb = \
                        hex2rgb(self.field_colors["Mark"])
                    self.vor_meshes[site_number].color.a = alpha_value
                    self.vor_lines[site_number].line.width = 3
                    self.vor_lines[site_number].color.rgb = \
                        hex2rgb(self.field_line_colors["Mark"])
                else:
                    self.vor_meshes[site_number].color.rgb = [1, 1, 1]
                    self.vor_lines[site_number].line.width = 1.5
                    self.vor_lines[site_number].color.rgb = \
                        [0.435, 0.51, 0.541]  # xkcd:steel grey

        # Handle visibility of sites
        self.on_hide_field("field_selection")

    def on_hide_field(self, _event, site_number=None):
        """
        Hide individual sites or specific collections of sites from view in
        the map-wide GUI.
        Helpful to declutter overview, e.g. eliminate non-responsive sites,
        sites categorized as non-A1 fields, etc.
        """
        if self.sites is None:  # Map isn't loaded
            return

        # Permit individual Site triggers without looping through all sites.
        if site_number is not None:
            for field in self.fields:
                if self.marks_active:  # Ignore fields and only hide Marked's
                    if field != "Mark":
                        continue
                elif field == "Mark":  # Ignore Marked's and only hide fields
                    continue

                if site_number in self.map_sets[field]:
                    if self.toggle_hide_dict[field].state == "down":
                        self.plot_dict[site_number].active = False
                        self.plot_dict[site_number].opacity = 0
                        self.vor_meshes[site_number].color.a = 0
                        self.vor_lines[site_number].color.a = 0
                        self.vor_active[site_number] = False

        else:  # Loop through all Sites
            for site in self.sites:
                site_number = site["number"]
                # Start by displaying all figures. Allows mixture of 
                # hiding/displaying sites that only belong to Field or Mark,
                # but not both.
                self.vor_meshes[site_number].color.a = \
                    self.field_alpha_slider.value_normalized
                self.vor_lines[site_number].color.a = 1
                self.vor_active[site_number] = True
                # Only show figures mass-hidden by Toggles.
                # Ignore site-specific figs ('Hide Figure')
                if not self.plot_dict[site_number].manually_hidden:
                    self.plot_dict[site_number].active = True
                    self.plot_dict[site_number].opacity = 1
                for field in self.fields:
                    if site_number in self.map_sets[field]:
                        if self.marks_active and field != "Mark":
                            continue
                        elif not self.marks_active and field == "Mark":
                            continue

                        if self.toggle_hide_dict[field].state == "down":
                            self.plot_dict[site_number].active = False
                            self.plot_dict[site_number].opacity = 0
                            self.vor_meshes[site_number].color.a = 0
                            self.vor_lines[site_number].color.a = 0
                            self.vor_active[site_number] = False

    def _redraw_all_plots(self, **re_plot_kwargs):
        """Re-render every overview SitePlot with the same kwargs."""
        for plot in self.plot_dict.values():
            plot.re_plot(**re_plot_kwargs)
            plot.figure_canvas.draw()

    def _on_plot_flag_toggle(self, toggle):
        """Shared handler for the map-wide TC display toggles."""
        if self._syncing_plot_flag_toggles:
            return

        attr = self._plot_flag_toggles[toggle]
        value = (toggle.state == "down")

        if attr == "use_lineplot" and value:
            self._syncing_plot_flag_toggles = True
            self.toggle_heatmap.state = "normal"
            self._syncing_plot_flag_toggles = False
        elif attr == "use_heatmap" and value:
            self._syncing_plot_flag_toggles = True
            self.toggle_lineplot.state = "normal"
            self._syncing_plot_flag_toggles = False

        for plot in self.plot_dict.values():
            if attr == "use_lineplot":
                plot.use_lineplot = value
                if value:
                    plot.use_heatmap = False
            elif attr == "use_heatmap":
                plot.use_heatmap = value
                if value:
                    plot.use_lineplot = False
            else:
                setattr(plot, attr, value)

        self.map_bubble_slider.disabled = (
            self.toggle_lineplot.state == "down" or
            self.toggle_heatmap.state == "down"
        )
        self._redraw_all_plots()

    def on_psth_ylim(self, _spinner, text):
        """Changing PSTH ylim's. Useful to emphasize weakly responsive sites."""
        self._redraw_all_plots(min_y=None if text == "None" else int(text))

    def export_map(self, _event):
        """Save Auditory Field selections and Marked sites to .json file."""
        if self.map_loaded:
            import datetime

            if self.marks_active:  # Save marks instead of fields
                for site in self.sites:
                    site_number = site["number"]
                    if site_number in self.map_sets["Mark"]:
                        marked = True
                    else:
                        marked = False
                    self.densetc_analysis_collection.update_one(
                        {"analysis_id": self.analysis_id, 
                         "number": site_number},
                        {"$set": {
                            "marked": marked
                        }})

            for field, map_set in self.map_sets.items():
                if field == "Mark":  # Don't save Marks as a field assignment!
                    continue
                for site_number in map_set:
                    self.densetc_analysis_collection.update_one(
                        {"analysis_id": self.analysis_id, 
                         "number": site_number},
                        {"$set": {
                            "field_assignment": field
                        }})

            # Update last_modified field on analysis_metadata
            today = str(datetime.datetime.now())
            self.analysis_metadata_collection.update_one(
                {"_id": self.analysis_id},
                {"$set": {
                    "last_modified": today
                }})

            InfoPopup("Success", "Fields / Marks saved!").open()

    def increase_figsize(self, _event):
        """Increase matplotlib figure size."""
        for fig in self.plot_dict.values():
            fig.size = (fig.width / 0.75, fig.height / 0.75)

    def decrease_figsize(self, _event):
        """Decrease matplotlib figure size."""
        for fig in self.plot_dict.values():
            fig.size = (fig.width * 0.75, fig.height * 0.75)

    def _build_ic_pseudo_sites(self):
        """
        IC recordings have a depth per site but no spatial map. Fabricate a
        two-column pseudo-voronoi: odd-numbered sites in the left column,
        even in the right, ordered top-to-bottom by depth so shallow
        (low-frequency) IC sits at the top.

        Cell heights come from inter-site depth spacing, which usually
        alternates ~0, ~200, ~0, ~200 (two sites per depth). We take the
        per-depth max to skip the zeros, then backfill the topmost site(s)
        which have nothing before them to diff against.
        """
        import pandas as pd
        ic_sites = [
            {"number": int(s["number"]), "depth": int(s["depth"])}
            for s in self.densetc_data_collection.find({})
        ]
        df = pd.DataFrame(ic_sites).sort_values("depth").reset_index(drop=True)

        odd_x, even_x = 0.25, 0.75
        df["x"] = df["number"].apply(lambda n: odd_x if n % 2 else even_x)
        df["y"] = df["depth"]

        df["inter_depth"] = df["y"].diff()
        # Two sites per depth → diff alternates 0, ~200. Max-per-depth skips
        # the zeros while tolerating the odd depth with only one site.
        df["inter_depth"] = df["y"].apply(
            lambda y: df.loc[df["y"] == y, "inter_depth"].max())
        # Top site(s) have no prior depth to diff → 0 → NaN → backfill.
        df.loc[df["inter_depth"] == 0, "inter_depth"] = np.nan
        df["inter_depth"] = df["inter_depth"].bfill()

        df["vert_up"] = df["y"] - df["inter_depth"] / 2
        df["vert_down"] = df["y"] + df["inter_depth"] / 2

        # Flip (depth increases downward on the probe but we want shallow at
        # the top of the display) then normalize to [0, 1].
        df[["y", "vert_up", "vert_down"]] *= -1
        lo, hi = df["vert_down"].min(), df["vert_up"].max()
        df[["y", "vert_up", "vert_down"]] = (
            (df[["y", "vert_up", "vert_down"]] - lo) / (hi - lo))

        sites = []
        for s in df.to_dict("records"):
            s["voronoi_centroid"] = [s["x"], s["y"]]
            if s["x"] == odd_x:
                s["voronoi_vertices"] = [
                    (0,   s["vert_down"]), (0,   s["vert_up"]),
                    (0.5, s["vert_up"]),   (0.5, s["vert_down"])]
            else:
                s["voronoi_vertices"] = [
                    (0.5, s["vert_down"]), (0.5, s["vert_up"]),
                    (1,   s["vert_up"]),   (1,   s["vert_down"])]
            sites.append(s)
        return sites

    def display_map(self):
        """Generate map visuals progressively so the window shows progress."""
        self.map_canvas.bind(size=self.update_line)
        self.map_canvas.bind(size=self.update_mesh)
        self.map_canvas.height = int(self.map_metadata["map_height"])
        self.map_canvas.width = int(self.map_metadata["map_width"])
        self.map_loaded = False
        self._set_loading_state(True)
        self._site_render_iter = iter(self.sites)
        self._total_sites = len(self.sites)
        self._rendered_sites = 0
        Clock.schedule_once(self._start_display_map_batches, 0)

    def _display_map_batch(self, _dt):
        rendered_this_tick = 0
        while rendered_this_tick < self._render_batch_size:
            try:
                site = next(self._site_render_iter)
            except StopIteration:
                self._finish_display_map()
                return False
            self._display_site(site)
            rendered_this_tick += 1

        self._update_load_status()
        Clock.schedule_once(self._display_map_batch, 0.01)
        return False

    def _finish_display_map(self):
        self._site_render_iter = None
        self.map_loaded = True
        self._set_loading_state(False)
        self._hide_load_popup()
        self._schedule_center_scroll_view()
        print("\n *** Ready! *** \n")

    def _display_site(self, site):
        # xy coords are already normalized, but here we reduce them to 90%
        # to provide some padding at the border of MapLayout -> allows
        # the user to move edge sites a little closer to the center for
        # easier viewing. Purely aesthetic.
        reduced_scale = [0.1, 0.9]
        site_number = site["number"]
        site_analysis = self.densetc_analysis[site_number]
        if "marked" not in site_analysis:
            site_analysis["marked"] = False

        if site_analysis["field_assignment"]:
            self.map_sets[site_analysis["field_assignment"]].add(site_number)
        if site_analysis["marked"]:
            self.map_sets["Mark"].add(site_number)

        x = (site["voronoi_centroid"][0] *
             (reduced_scale[1] - reduced_scale[0]) /
             (1 - 0) + reduced_scale[0])
        y = (site["voronoi_centroid"][1] *
             (reduced_scale[1] - reduced_scale[0]) /
             (1 - 0) + reduced_scale[0])

        model = SiteModel(site_number,
                          self.densetc_data[site_number],
                          site_analysis,
                          self.stim_config)
        self.site_models[site_number] = model
        site_plot = SitePlot(
            model=model,
            detailed_plot=False,
            is_ic=self.ic_bool,
            cf_cmap=self.cf_colormap_dropdown.text,
            heatmap_cmap=self.heatmap_colormap_dropdown.text,
            size_hint=(None, None),
            pos_hint={"center_x": x, "center_y": y},
            height=150,
            width=200)

        self.plot_dict[site_number] = site_plot
        self.map_canvas.add_widget(site_plot)
        with self.map_canvas.canvas.before:
            if site_analysis["field_assignment"] and not self.marks_active:
                line_color = Color(*hex2rgb(
                    self.field_line_colors[site_analysis["field_assignment"]]))
                lw = 3
            elif site_analysis["marked"] and self.marks_active:
                line_color = Color(*hex2rgb(self.field_line_colors["Mark"]))
                lw = 3
            else:
                line_color = Color(0.435, 0.51, 0.541, 1)
                lw = 1.5

            poly_norm_points = site["voronoi_vertices"]
            poly_x = [pnt[0] * (reduced_scale[1] - reduced_scale[0]) /
                      (1 - 0) + reduced_scale[0] for pnt in poly_norm_points]
            poly_y = [pnt[1] * (reduced_scale[1] - reduced_scale[0]) /
                      (1 - 0) + reduced_scale[0] for pnt in poly_norm_points]
            height = self.map_canvas.height
            width = self.map_canvas.width
            poly_x_adjusted = list(np.array(poly_x) * width)
            poly_y_adjusted = list(np.array(poly_y) * height)
            adjusted_points = list(itertools.chain(*zip(poly_x_adjusted,
                                                        poly_y_adjusted)))
            line_ = Line(points=adjusted_points, width=lw, close=True)
            self.vor_lines[site_number] = LineTuple(
                line=line_, color=line_color, x_norm=poly_x, y_norm=poly_y,
                site_number=site_number)
            mesh_adjusted_points = list(itertools.chain(*[
                (x, y, 0, 0) for x, y in zip(poly_x_adjusted, poly_y_adjusted)
            ]))
            indices = list(range(len(poly_x_adjusted)))

            if site_analysis["field_assignment"] and not self.marks_active:
                mesh_color = Color(*hex2rgb(
                    self.field_colors[site_analysis["field_assignment"]]))
            elif site_analysis["marked"] and self.marks_active:
                mesh_color = Color(*hex2rgb(self.field_colors["Mark"]))
            else:
                mesh_color = Color(1, 1, 1, 1)

            mesh_ = Mesh(vertices=mesh_adjusted_points, indices=indices,
                         mode="triangle_fan")
            self.vor_meshes[site_number] = MeshTuple(
                mesh=mesh_, color=mesh_color, x_norm=poly_x, y_norm=poly_y,
                site_number=site_number)
            self.vor_active[site_number] = True

        self._rendered_sites += 1

    def change_bubble_size(self, _slider, value):
        """Event signaling update to max bubble size for TC plot."""
        for plot in self.plot_dict.values():
            plot.max_bubble_size = value
            plot.update_bubble_size()
            plot.figure_canvas.draw()

    def change_field_alpha(self, slider, _value):
        """Event signaling update to alpha values for field colors."""
        value = slider.value_normalized
        for mesh_tuple in self.vor_meshes.values():
            site_number = mesh_tuple.site_number
            if self.vor_active[site_number]:
                mesh_tuple.color.a = value

    def flash_cell(self, map_number, unsaved_changes=False, marked=False):
        """Flash voronoi cell of Site-screen user nagivated away from."""
        # First add/remove site from Mark map_set
        if marked:
            self.map_sets["Mark"].add(map_number)
        else:
            try:
                self.map_sets["Mark"].remove(map_number)
            except KeyError:  # If site is not in set, skip
                pass

        self.flash_times = 0
        self.flash_line = self.vor_lines[map_number]
        self.flash_mesh = self.vor_meshes[map_number]
        alpha_value = self.field_alpha_slider.value_normalized
        if unsaved_changes:
            self.flash_lw = 5
            self.flash_line_color = hex2rgb(self.unsaved_line_color)
            self.flash_mesh_color = hex2rgb(self.unsaved_mesh_color)
            self.flash_mesh_color[3] = alpha_value
        else:
            field_assigned = False
            for field in self.fields:
                if self.marks_active:  # Ignore auditory fields
                    if field != "Mark":
                        continue
                elif field == "Mark":  # Ignore if marks marks_active is False
                    continue

                if map_number in self.map_sets[field]:
                    field_assigned = True
                    self.flash_lw = 3
                    self.flash_line_color = hex2rgb(
                        self.field_line_colors[field])
                    self.flash_mesh_color = hex2rgb(self.field_colors[field])
                    self.flash_mesh_color[3] = alpha_value

            if not field_assigned:
                self.flash_lw = 1.5
                self.flash_line_color = [0.435, 0.51, 0.541]  # xkcd:steel grey
                self.flash_mesh_color = [1, 1, 1, alpha_value]

        self.flash_clock_event = Clock.schedule_interval(self.flash_callback, 
                                                         0.08)

    def flash_callback(self, _dt):
        """Simple callback to flash voronoi cell 8x, then canceling clock."""
        self.flash_times = self.flash_times + 1
        if self.flash_times <= 8:
            if (self.flash_times % 2) == 1:
                self.flash_line.line.width = 5
                # xkcd:cherry red
                self.flash_line.color.rgb = hex2rgb("#f7022a")
                # xkcd:almost black
                self.flash_mesh.color.rgb = hex2rgb("#070d0d")
            else:
                self.flash_line.line.width = self.flash_lw
                self.flash_line.color.rgb = self.flash_line_color
                self.flash_mesh.color.rgba = self.flash_mesh_color
        else:
            self.flash_clock_event.cancel()

    def update_line(self, _layout_instance, size):
        """Update Kivy canvas lines when user resizes GUI."""
        width, height = size
        for line_tuple in self.vor_lines.values():
            poly_x_adjusted = list(np.array(line_tuple.x_norm) * width)
            poly_y_adjusted = list(np.array(line_tuple.y_norm) * height)
            adjusted_points = list(itertools.chain(*zip(poly_x_adjusted, 
                                                        poly_y_adjusted)))
            line_tuple.line.points = adjusted_points

    def update_mesh(self, _layout_instance, size):
        """Update Kivy canvas meshes when user resizes GUI."""
        width, height = size
        for mesh_tuple in self.vor_meshes.values():
            poly_x_adjusted = list(np.array(mesh_tuple.x_norm) * width)
            poly_y_adjusted = list(np.array(mesh_tuple.y_norm) * height)
            mesh_adjusted_points = list(itertools.chain(*[
                (x, y, 0, 0) for x, y in 
                zip(poly_x_adjusted, poly_y_adjusted)]))
            mesh_tuple.mesh.vertices = mesh_adjusted_points


class MapLayout(FloatLayout):
    def __init__(self, *, gui, **kwargs):
        """
        `gui` is the owning FieldSelectionGUI. Stored so touch handlers can
        reach voronoi state, toggles, and plot dicts without walking
        `.parent.parent` through the intervening ScrollView (allows
        rearranging layout nesting without breaking chains).
        """
        super().__init__(**kwargs)
        self.gui = gui

    def on_touch_down(self, touch):
        """
        Double-tap on a voronoi cell opens that site's detail screen.

        Otherwise: if a paint toggle is active, start a red stroke that
        on_touch_up will resolve into cell selections. If no paint toggle is
        active, defer to FloatLayout so the ScrollView can handle panning.

        Hit-testing rebuilds each polygon from its current Kivy Line points
        every time, since zoom rescales absolute coordinates.
        """
        gui = self.gui
        if touch.is_double_tap:
            for line_ in gui.vor_lines.values():
                poly_points = line_.line.points
                poly_xy = list(zip(poly_points[0::2], poly_points[1::2]))
                if MplPath(poly_xy).contains_point((touch.x, touch.y)):
                    gui.open_site_screen(line_.site_number)
        if gui.paint_mode_active:
            with self.canvas:
                Color(1, 0, 0)
                touch.ud["line"] = Line(points=(touch.x, touch.y), width=1.5)
        else:
            super().on_touch_down(touch)

    def on_touch_move(self, touch):
        """
        Extend the paint stroke.
        """
        try:
            if self.gui.paint_mode_active:
                touch.ud["line"].points += [touch.x, touch.y]
        except KeyError:
            # Thrown if program tries to interpret line drawn over GUI elements
            super().on_touch_move(touch)
            pass

    def on_touch_up(self, touch):
        """
        Three paths:

        1. Mouse-wheel → zoom. Kivy delivers wheel events to ScrollView
           children as touch_up, which is why zoom lives here rather than
           on MapScroll.
        2. Paint stroke finished → hit-test the stroke against every cell
           and apply the active toggle's action to each cell touched.
        3. Neither → let Kivy handle it.
        """
        if touch.is_mouse_scrolling:
            w, h = self.width, self.height
            if touch.button == "scrollup":
                self.size = (w * 0.9, h * 0.9)
            elif touch.button == "scrolldown":
                self.size = (w * 1.1, h * 1.1)
            return True

        gui = self.gui

        if not gui.paint_mode_active:
            super().on_touch_up(touch)
            return

        try:
            stroke = touch.ud["line"].points
            selection_points = list(zip(stroke[0::2], stroke[1::2]))
        except KeyError:
            # Thrown if program tries to interpret line drawn over GUI elements
            super().on_touch_up(touch)
            return

        alpha = gui.field_alpha_slider.value_normalized

        for line_ in gui.vor_lines.values():
            pts = line_.line.points
            poly = list(zip(pts[0::2], pts[1::2]))
            if not MplPath(poly).contains_points(selection_points).any():
                continue

            site_num = line_.site_number
            if not gui.vor_active[site_num]:
                # Cell is currently hidden → non-interactive.
                continue

            if gui.toggle.state == "down":
                # --- Select: assign the active field/Mark to this site ---
                line_.line.width = 3
                chosen = gui.field_spinner.text
                for field in gui.fields:
                    if field == chosen:
                        gui.map_sets[field].add(site_num)
                        gui.vor_meshes[site_num].color.rgb = \
                            hex2rgb(gui.field_colors[field])
                        gui.vor_meshes[site_num].color.a = alpha
                        line_.color.rgb = \
                            hex2rgb(gui.field_line_colors[field])
                        # If Hide-<field> is toggled, re-evaluate visibility
                        # now that this site belongs to it — lets the user
                        # progressively select-then-hide. The string arg is
                        # a dummy to satisfy the Kivy callback signature.
                        gui.on_hide_field("field_selection",
                                          site_number=site_num)
                    else:
                        # Sites belong to exactly one auditory field, but
                        # skip if user is marking them instead
                        if gui.marks_active or field == "Mark":
                            continue
                        gui.map_sets[field].discard(site_num)

            elif gui.deselect_toggle.state == "down":
                # --- Deselect: strip assignment, repaint as blank ---
                line_.color.rgb = [0.435, 0.51, 0.541]  # xkcd:steel grey
                line_.line.width = 1.5
                gui.vor_meshes[site_num].color.rgb = [1, 1, 1]
                gui.vor_meshes[site_num].color.a = alpha
                for field in gui.fields:
                    # Same as selection rule above
                    if field == "Mark" and not gui.marks_active:
                        continue
                    if field != "Mark" and gui.marks_active:
                        continue
                    gui.map_sets[field].discard(site_num)

            elif gui.show_figure_toggle.state == "down":
                plot = gui.plot_dict[site_num]
                plot.active = True
                plot.opacity = 1
                plot.manually_hidden = False

            elif gui.hide_figure_toggle.state == "down":
                plot = gui.plot_dict[site_num]
                plot.active = False
                plot.opacity = 0
                plot.manually_hidden = True

        self.canvas.remove(touch.ud["line"])


class MapScroll(ScrollView):
    def __init__(self, **kwargs):
        """Overrides `on_touch_down` of `ScrollView` to permit zooming."""
        super(MapScroll, self).__init__(**kwargs)

    def on_touch_down(self, touch):
        """
        Override mousewheel scrolling events in order to allow zooming with the
        mousewheel (normal ScrollView simply scrolls around with mousewheel, 
        and cannot zoom)
        """
        if touch.is_mouse_scrolling:
            # passes on the event to other GUI components (allows zooming 
            # instead of scrolling)
            return True
        else:
            # if not mousewheel, consume input like normal
            super(MapScroll, self).on_touch_down(touch)


class SitePlot(RelativeLayout):
    """
    PSTH + TC rendering for one site.

    Map instances share one SiteModel.
    All analysis state and derived
    arrays live on the model; this class is rendering and mouse
    handling only.
    """
    def __init__(self, model, detailed_plot, is_ic,
                 cf_cmap, heatmap_cmap, **layout_kwargs):
        super().__init__(**layout_kwargs)
        self.model = model
        self.detailed_plot = detailed_plot
        cfg = model.config

        if detailed_plot:
            # Listen for signals
            import blinker
            self.on_changes_signal = blinker.Signal()
            self.on_cf_pick_signal = blinker.Signal()

        # Allow user to change cmaps used for plots
        self.cf_cmap = mpl_colormaps[cf_cmap]
        self.heatmap_cmap = heatmap_cmap
        # TODO test 48khz
        self.norm = mpl_colors.Normalize(
            vmin=0, vmax=cfg.num_frequency - 1)

        # IC responses are faster than cortical, so the latency colour
        # range is tighter and lower. PowerNorm(0.65) stretches the low
        # end where most onsets sit.
        lo, hi = (1, 16) if is_ic else (5, 20)
        self.speed_cmap = cmocean.cm.speed
        self.speed_norm = mpl_colors.PowerNorm(
            0.65, vmin=lo, vmax=hi)

        # Display toggles
        self.use_smooth_tc = False
        self.use_lineplot = False
        self.use_heatmap = False
        self.use_contour = False
        self.use_bw = True
        self.manually_hidden = False
        if detailed_plot:
            self.active = False
            self.bin_size = 1
            self.max_bubble_size = 30
        else:
            self.active = True
            self.bin_size = 5
            self.max_bubble_size = 6

        # Artist handles, populated on render
        self.bubble = None
        self.line = None
        self.heatmap = None
        self.psth = None
        self.latency_txt = None
        self.cf_marker = None
        self.contour_line = None
        self.sdf_line = None
        self.spont_line = None
        self.onset_line = None
        self.offset_line = None
        self.bw_lines = {lvl: None for lvl in cfg.bw_levels_db}
        self.bw_markers = {lvl: [None, None] for lvl in cfg.bw_levels_db}

        # Interaction flags
        self.onset_press = False
        self.offset_press = False
        self.picking_cf = False
        self.bw_pressed = False
        self.bw_press = {lvl: [False, False] for lvl in cfg.bw_levels_db}

        # Last-rendered TC values, kept so update_bubble_size() can
        # rescale without refetching the array.
        self.row = self.col = self.val = np.array([])

        self.fig = Figure()
        self.ax = [
            self.fig.add_axes([0.125, 0.495, 0.775, 0.385]),
            self.fig.add_axes([0.125, 0.11, 0.775, 0.385]),
        ]

        # Aesthetics
        self.fig.patch.set_alpha(0)
        self._has_render = False
        self._needs_redraw = detailed_plot
        
        if not detailed_plot:
            self.bubble_plot()
            self.psth_plot()
            self._has_render = True
            self._needs_redraw = False
        else:
            self.ax[0].axis("off")
            self.ax[1].axis("off")

        # Generate Kivy widget for displaying in GUI
        self.figure_canvas = FigureCanvas(self.fig)

        # Attach mouse events to the latency lines so user can move them.
        self.fig.canvas.mpl_connect("button_press_event", 
                                    self.mouse_click_event)
        self.fig.canvas.mpl_connect("motion_notify_event", 
                                    self.mouse_move_event)
        self.fig.canvas.mpl_connect("button_release_event", 
                                    self.mouse_release_event)

        self.add_widget(self.figure_canvas)

    def mark_dirty(self):
        """Queue a full redraw for the next time the plot is shown."""
        self._needs_redraw = True

    def draw_canvas(self):
        self.figure_canvas.draw()
        self._has_render = True
        self._needs_redraw = False

    def ensure_rendered(self, axis_visible="off", min_y=None):
        if self._needs_redraw or not self._has_render:
            self.re_plot(axis_visible=axis_visible, min_y=min_y)
            self.draw_canvas()

    # -- shortcuts ------------------------------------------------------
    @property
    def st(self):
        """Working analysis state (model.working). All edits go here."""
        return self.model.working

    @property
    def bubble_color(self):
        cf = self.st.cf_idx
        return "r" if cf is None else self.cf_cmap(self.norm(cf))

    @property
    def lat_color(self):
        # Non-responsive sites (no CF) get a sentinel colors for TC and PSTH.
        if self.st.cf_idx is None:
            return "m"
        return self.speed_cmap(self.speed_norm(self.st.onset))
    
    def _current_tc(self):
        """
        TC array for the active rendering, respecting use_smooth_tc.
        Memoised on (onset, offset) inside the model, so calling this on
        every redraw is free unless latencies moved.
        """
        if self.use_smooth_tc:
            return self.model.ttest_tc()
        return self.model.raw_tc()

    @staticmethod
    def _scale_sizes(vals, max_size):
        """
        Scale to [0, max_size], appending a 0 so the full range is used
        (otherwise the lowest spike count maps to size 0). Returns the
        input unchanged if it's empty.
        """
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return vals
        vmax = vals.max(initial=0)
        if vmax <= 0:
            return np.zeros_like(vals, dtype=float)
        return vals * (max_size / vmax)
    
    # -- full renders ---------------------------------------------------
    def re_plot(self, axis_visible="off", min_y=None):
        """Re-plot TC."""
        if self.use_lineplot:
            self.line_plot(axis_visible=axis_visible)
        elif self.use_heatmap:
            self.heatmap_plot(axis_visible=axis_visible)
        else:
            self.bubble_plot(axis_visible=axis_visible)
        self.psth_plot(min_y=min_y)

    def re_color(self, cf_cmap="viridis", heatmap_cmap="inferno"):
        """Update bubble plot or heatmap colors."""
        self.heatmap_cmap = heatmap_cmap
        self.cf_cmap = mpl_colormaps[cf_cmap]
        # TODO allow user to change No CF color (default is red)
        if self.use_heatmap and self.heatmap is not None:
            self.heatmap.set_cmap(self.heatmap_cmap)
        elif not self.use_lineplot and self.bubble is not None:
            self.bubble.update({"facecolors": self.bubble_color})

    def bubble_plot(self, axis_visible="off", axis_color="xkcd:white"):
        ax = self.ax[1]
        cfg = self.model.config
        tc = self._current_tc()
        self.row, self.col = np.where(tc > 0)
        self.val = tc[self.row, self.col]

        if self._has_render:
            ax.clear()
        scaled = self._scale_sizes(self.val, self.max_bubble_size)
        self.bubble = ax.scatter(x=self.col, y=self.row, s=scaled ** 2,
                                 edgecolors="black", lw=0.5,
                                 color=self.bubble_color)
        ax.set_facecolor(axis_color)
        self._draw_tc_overlays(ax)
        ax.set_xlim([0, cfg.num_frequency])
        ax.set_ylim([0, cfg.num_intensity])
        ax.axis(axis_visible)

    def line_plot(self, axis_visible="on", axis_color="xkcd:black"):
        """Matches older analysis plotting style (tc_exploror, iykyk)."""
        ax = self.ax[1]
        cfg = self.model.config
        max_len = 1
        tc = self._current_tc()
        self.row, self.col = np.where(tc > 0)
        self.val = tc[self.row, self.col]

        ax.clear()
        scaled = self._scale_sizes(self.val, max_len)
        segs = [[[x, y + 0.25], [x, y + 0.25 - s]]
                for x, y, s in zip(self.col, self.row, scaled)]
        self.line = LineCollection(segs, linewidths=2, colors="y")
        ax.add_collection(self.line)
        ax.set_facecolor(axis_color)
        self._draw_tc_overlays(
            ax, contour_color="xkcd:white" if self.detailed_plot else None)
        ax.set_xlim([0, cfg.num_frequency])
        ax.set_ylim([0, cfg.num_intensity])
        ax.axis(axis_visible)

    def heatmap_plot(self, axis_visible="on"):
        """For fun and profit (and heat)."""
        ax = self.ax[1]
        cfg = self.model.config
        tc = self._current_tc()

        ax.clear()
        self.heatmap = ax.imshow(tc, cmap=self.heatmap_cmap, aspect="auto")
        self._draw_tc_overlays(
            ax, contour_color="xkcd:white" if self.detailed_plot else None)
        ax.set_xlim([0, cfg.num_frequency - 1])
        ax.set_ylim([0, cfg.num_intensity - 1])
        ax.axis(axis_visible)

    def _draw_tc_overlays(self, ax, contour_color=None):
        """
        CF marker, BW lines/markers, and contour on top of whichever TC
        rendering just populated `ax`. Dark-background renderings pass
        a white contour_color; None uses matplotlib's default cycle.
        """
        cfg = self.model.config
        st = self.st

        if self.use_bw and st.cf_idx is not None:
            for lvl in cfg.bw_levels_db:
                idx = st.bw_idx[lvl]
                if idx[0] is None:
                    continue
                y = st.thresh_idx + cfg.bw_row_offset(lvl)
                self.bw_lines[lvl] = ax.plot(idx, [y, y], "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw_markers[lvl][0] = ax.plot(
                        idx[0], y, "rd", ms=8, picker=5)[0]
                    self.bw_markers[lvl][1] = ax.plot(
                        idx[1], y, "rd", ms=8, picker=5)[0]

        if self.use_contour:
            contour = self.model.contour_tc()
            if contour_color:
                self.contour_line = ax.contour(contour, levels=[0],
                                               colors=contour_color)
            else:
                self.contour_line = ax.contour(contour, levels=[0])

        if st.cf_idx is not None:
            self.cf_marker = ax.plot(st.cf_idx, st.thresh_idx,
                                     "r*", ms=8, alpha=0.5)[0]

    def psth_plot(self, min_y=None):
        ax = self.ax[0]
        cfg = self.model.config
        st = self.st
        raw = self.model.raw_psth
        sweep = cfg.sweep_length_ms
        bin_size = self.bin_size

        if self._has_render:
            ax.clear()
        if bin_size in (1, 5):
            num_bins = round((sweep - 1) / bin_size)
            binned, bin_edges = np.histogram(
                range(len(raw)), bins=num_bins, weights=raw)
        else:
            binned = raw
            num_bins = len(binned)
            bin_edges = np.arange(num_bins + 1)

        hist_peak = int(np.argmax(binned))
        # Quick visual peak rate (not the driven rate), and it changes
        # with bin size. Just a label, not an analysis output.
        # TODO Clarity and explicitness on the user end is a good thing to aim for
        ms_mult = 1000 // bin_size
        peak_rate = int(round((binned[hist_peak] * ms_mult) / cfg.num_tones))

        self.psth = ax.stairs(
            binned,
            bin_edges,
            baseline=0,
            fill=True,
            color=self.lat_color,
            edgecolor="#fdfdfe",
            linewidth=0.4,
        )

        if self.detailed_plot:
            self.sdf_line = ax.plot(
                self.model.sdf * bin_size * cfg.num_tones,
                lw=2, color="xkcd:amber")[0]

        # If a minimum max y-lim value is set (so small Hz do indeed look 
        # small), set it IF it is larger than current
        if min_y:
            min_y_counts = (min_y / ms_mult) * cfg.num_tones
            if ax.get_ylim()[1] < min_y_counts:
                ax.set_ylim([0, min_y_counts])
                y_val = min_y_counts
            else:
                y_val = binned[hist_peak]
        else:
            y_val = binned[hist_peak]

        cf_val = ("-" if st.cf_idx is None
                  else f"{cfg.frequencies_hz[st.cf_idx] / 1000:.1f}")
        self.latency_txt = ax.annotate(
            f"On: {st.onset}, Pk: {st.peak}, Off: {st.offset}\n"
            f"Rate: {peak_rate} Hz, CF: {cf_val} kHz",
            (1.25, 0), xytext=[st.offset + 5, y_val],
            size=10, va="top", name="Segoe UI", weight="bold",
            color="xkcd:dark blue")
        ax.set_xlim([0, sweep - 1])

        # If detailed plot, plot spontaneous and SDF
        if self.detailed_plot:
            # Spont was calculated at 1ms bin size
            spont = (self.model.spont_rate / 1000) * bin_size * cfg.num_tones
            self.spont_line = ax.plot([0,sweep-1], [spont,spont], "c", lw=2)[0]

        # Plot latency lines on psth
        lat_lw = 3 if self.detailed_plot else 1
        self.onset_line = ax.plot([st.onset, st.onset], [0, y_val],
                                  "r", lw=lat_lw, picker=2)[0]
        self.offset_line = ax.plot([st.offset, st.offset], [0, y_val],
                                   "r", lw=lat_lw, picker=2)[0]
        ax.axis("off")

    # -- partial updates during latency drag ----------------------------
    def update_bubble(self):
        tc = self._current_tc()
        self.row, self.col = np.where(tc > 0)
        self.val = tc[self.row, self.col]
        scaled = self._scale_sizes(self.val, self.max_bubble_size)
        offsets = np.column_stack((self.col, self.row))
        self.bubble.update({"offsets": offsets, "sizes": scaled ** 2})

    def update_bubble_size(self):
        """Rescale existing bubbles without refetching the TC."""
        scaled = self._scale_sizes(self.val, self.max_bubble_size)
        self.bubble.update({"sizes": scaled ** 2})

    def update_line(self):
        max_len = 1
        tc = self._current_tc()
        self.row, self.col = np.where(tc > 0)
        self.val = tc[self.row, self.col]
        scaled = self._scale_sizes(self.val, max_len)
        segs = [[[x, y + 0.25], [x, y + 0.25 - s]]
                for x, y, s in zip(self.col, self.row, scaled)]
        self.line.set_segments(segs)

    def update_heatmap(self):
        self.heatmap.set_data(self._current_tc())

    # -- mouse handling -------------------------------------------------
    def mouse_click_event(self, event):
        event.x, event.y = self.to_window(*self.to_parent(event.x, event.y))
        if not self.active:
            return
        if self.detailed_plot:
            # event.inaxes is unreliable after the coord transform above
            # (it's computed pre-transform and can be None near axis
            # edges). Transform into each axes' unit coords instead.
            x0, y0 = self.ax[0].transAxes.inverted().transform(
                [event.x, event.y])
            x1, y1 = self.ax[1].transAxes.inverted().transform(
                [event.x, event.y])
            if 0 <= x0 <= 1 and 0 <= y0 <= 1:
                self.on_pick_line(event)
            elif 0 <= x1 <= 1 and 0 <= y1 <= 1:
                if self.picking_cf:
                    self.pick_cf(event)
                elif self.use_bw:
                    self.pick_bw(event)
        else:
            # Overview axes coords don't line up with event coords even
            # after the transform, so let on_pick_line do its own
            # contains() check.
            self.on_pick_line(event)

    def mouse_move_event(self, event):
        """Drag latency or bandwidth lines on user interaction."""
        if self.onset_press or self.offset_press:
            x, y = self.to_window(*self.to_parent(event.x, event.y))
            self.move_line(x, y)
        elif self.active and self.bw_pressed:
            x, y = self.to_window(*self.to_parent(event.x, event.y))
            self.move_bw(x, y)

    def mouse_release_event(self, _event):
        """Finalize latency or bandwidth lines after user interaction."""
        if self.onset_press or self.offset_press:
            self.off_pick()
        elif self.active and self.bw_pressed:
            self.off_bw()

    def on_pick_line(self, event):
        lat_lw = 5 if self.detailed_plot else 1.5
        if self.onset_line.contains(event)[0]:
            self.onset_line.set_lw(lat_lw)
            self.onset_press = True
            event.canvas.draw()
        elif self.offset_line.contains(event)[0]:
            self.offset_line.set_lw(lat_lw)
            self.offset_press = True
            event.canvas.draw()

    def move_line(self, x, y):
        import analysis_functions as afunc
        xdata, _ = self.ax[0].transData.inverted().transform((x, y))
        if xdata is None:
            return
        cfg = self.model.config
        st = self.st

        # Onset/offset drags are identical apart from which line/field
        # they touch and the no-crossover constraint.
        if self.onset_press and 0 <= xdata <= cfg.sweep_length_ms \
                and xdata < self.offset_line.get_xdata()[0]:
            self.onset_line.set_xdata([xdata, xdata])
            st.onset = int(round(xdata))
        elif self.offset_press and 0 <= xdata <= cfg.sweep_length_ms \
                and xdata > self.onset_line.get_xdata()[0]:
            self.offset_line.set_xdata([xdata, xdata])
            st.offset = int(round(xdata))
        else:
            return

        if self.detailed_plot:
            self.on_changes_signal.send()
            raw = self.model.raw_psth
            peak_hist = raw.copy()
            peak_hist[:st.onset] = peak_hist[st.offset:] = 0
            st.peak = int(np.argmax(peak_hist))
            st.peak_driven_rate = afunc.get_peak_driven_rate(
                raw[st.onset:st.offset], self.model.spont_rate,
                cfg.num_tones)
            self.latency_txt.set_text(
                f"{st.onset}, {st.peak}, {st.offset}")

        if self.use_lineplot:
            self.update_line()
        elif self.use_heatmap:
            self.update_heatmap()
        else:
            self.update_bubble()
        self.figure_canvas.draw()

    def off_pick(self):
        """Final UX for user updating latency line."""
        lat_lw = 3 if self.detailed_plot else 1
        self.onset_line.set_lw(lat_lw)
        self.offset_line.set_lw(lat_lw)
        self.onset_press = self.offset_press = False
        if self.detailed_plot:
            # Full PSTH re-render to reposition the annotation and
            # refresh the rate text.
            self.psth_plot()
        self.figure_canvas.draw()

    def pick_cf(self, event):
        """
        User clicked the TC plot to set CF/threshold. Snap to the
        nearest grid index, then seed/clear BW handles according to
        which rows are still on the grid at the new threshold.
        """
        xdata, ydata = self.ax[1].transData.inverted().transform(
            (event.x, event.y))
        if xdata is None or ydata is None:
            return
        cfg = self.model.config
        st = self.st

        st.cf_idx = int(round(xdata))
        st.thresh_idx = int(round(ydata))
        if self.cf_marker is None:
            self.cf_marker = self.ax[1].plot(st.cf_idx, st.thresh_idx,
                                             "r*", ms=8, alpha=0.5)[0]
        else:
            self.cf_marker.set_xdata([st.cf_idx])
            self.cf_marker.set_ydata([st.thresh_idx])

        for lvl in cfg.bw_levels_db:
            row = st.thresh_idx + cfg.bw_row_offset(lvl)
            if row < cfg.num_intensity:
                if st.bw_idx[lvl][0] is None:
                    st.bw_idx[lvl] = [10, cfg.num_frequency - 10]
            else:
                st.bw_idx[lvl] = [None, None]

        self.picking_cf = False
        self.on_cf_pick_signal.send()
        self.on_changes_signal.send()

    def pick_bw(self, event):
        """Grab one BW marker for dragging."""
        for lvl in self.model.config.bw_levels_db:
            markers = self.bw_markers[lvl]
            if markers[0] is None:
                continue
            for side in (0, 1):
                if markers[side].contains(event)[0]:
                    self.bw_pressed = True
                    markers[side].set_ms(12)
                    self.bw_press[lvl][side] = True
                    event.canvas.draw()
                    return

    def move_bw(self, event_x, event_y):
        """Drag the held BW marker, clamped and non-crossing."""
        xdata, _ = self.ax[1].transData.inverted().transform(
            (event_x, event_y))
        if xdata is None:
            return
        cfg = self.model.config
        st = self.st
        max_idx = cfg.num_frequency - 1

        for lvl in cfg.bw_levels_db:
            press = self.bw_press[lvl]
            markers = self.bw_markers[lvl]
            idx = st.bw_idx[lvl]
            if press[0] and xdata < markers[1].get_xdata()[0]:
                x = max(0, int(round(xdata)))
                markers[0].set_xdata([x])
                idx[0] = x
                self.bw_lines[lvl].set_xdata(idx)
                if self.detailed_plot:
                    self.on_changes_signal.send()
            elif press[1] and xdata > markers[0].get_xdata()[0]:
                x = min(max_idx, int(round(xdata)))
                markers[1].set_xdata([x])
                idx[1] = x
                self.bw_lines[lvl].set_xdata(idx)
                if self.detailed_plot:
                    self.on_changes_signal.send()
        self.figure_canvas.draw()

    def off_bw(self):
        self.bw_pressed = False
        for lvl in self.model.config.bw_levels_db:
            self.bw_press[lvl] = [False, False]
            if self.bw_markers[lvl][0] is not None:
                self.bw_markers[lvl][0].set_ms(8)
                self.bw_markers[lvl][1].set_ms(8)
        self.figure_canvas.draw()

class InfoPopup(Popup):
    """
    One-button modal. Replaces messagebox.showinfo / showerror.

    Non-blocking — open() returns immediately. If you need to do something
    after the user acknowledges, bind to on_dismiss.
    """
    def __init__(self, title, message, **kwargs):
        layout = BoxLayout(orientation="vertical", padding=12, spacing=12)
        lbl = Label(text=message, halign="center", valign="middle")
        lbl.bind(size=lbl.setter("text_size"))  # enable text wrapping
        layout.add_widget(lbl)
        ok = Button(text="OK", size_hint=(1, None), height=44)
        ok.bind(on_release=self.dismiss)
        layout.add_widget(ok)
        super().__init__(title=title, content=layout,
                         size_hint=(0.45, 0.3), auto_dismiss=True, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Field Selection GUI. Normally launched from the "
                    "map_analysis CLI, which handles DB and analysis "
                    "selection through native dialogs and passes the "
                    "result here. To switch maps, close this window "
                    "and pick again from the CLI.")
    parser.add_argument("--db", required=True,
                        help="path to the subject's TinyDB JSON file")
    parser.add_argument("--analysis-id", required=True,
                        help="_id of the analysis_metadata doc to load")
    parser.add_argument("--ic", action="store_true",
                        help="treat as an inferior-colliculus pseudo-map "
                             "rather than a cortical map")
    args = parser.parse_args()
    FieldSelectionApp(args.db, args.analysis_id, args.ic).run()
