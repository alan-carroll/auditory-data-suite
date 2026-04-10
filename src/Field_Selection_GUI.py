# Disable Kivy's argument parser so the --db / --analysis-id / --ic
# flags reach argparse in __main__ instead of being consumed (and
# rejected) by Kivy at import time.
import os
os.environ["KIVY_NO_ARGS"] = "1"

import datetime
import itertools
import numpy as np
import matplotlib
# Pin non-interactive backend prior to importing kivy
matplotlib.use("Agg")
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
import matplotlib.pyplot as plt
# Pip-installable replacement for the legacy `garden install matplotlib` flower.
# Aliased so the rest of the module doesn't need to change.
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg as FigureCanvas
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
from matplotlib.path import Path as MplPath
import pandas as pd
import cmocean
from kivy.utils import get_color_from_hex as hex2rgb
from kivy.uix.screenmanager import Screen, ScreenManager
import logging
from kivy.uix.slider import Slider
import analysis_functions as afunc
import blinker
from tinymongo_fix.tinymongo_fix import TinyMongoClient
from sklearn.preprocessing import minmax_scale
from collections import namedtuple
from matplotlib.collections import LineCollection
import warnings
from matplotlib.axes._axes import _log as matplotlib_axes_logger

# dB-above-threshold levels at which bandwidths are measured. The y-offset
# on the TC plot for each is level // intensity_step (currently 5 dB steps,
# so BW10 sits 2 rows above threshold, BW20 sits 4 rows above, etc.).
BW_LEVELS = (10, 20, 30, 40)

# Ignore warnings about opening too many figures or not finding contour lines 
# issued by matplotlib
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib_axes_logger.setLevel('ERROR')

logging.basicConfig(filename="map_gui_log.log", filemode="w")
logging.getLogger('matplotlib.font_manager').disabled = True

Window.clearcolor = (1, 1, 1, 1)


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

        # Flash site when returning to Map screen
        self.flash_signal = blinker.signal("flash")

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

        # Initialize default bubble plot
        densetc_plot.max_bubble_size = 30
        densetc_plot.bubble_plot(axis_visible="on")
        self.densetc_plot = densetc_plot
        if self.densetc_plot.marked:
            self.mark_site_toggle.state = "down"
        self.layout.add_widget(self.densetc_plot)

        self.layout.add_widget(self.tools_layout)
        self.add_widget(self.layout)

        # Listen for user changes to analysis
        self.densetc_plot.on_changes_signal.connect(self.changes_made)
        self.densetc_plot.on_cf_pick_signal.connect(self.cf_picked)
        
    def redraw(self):
        """Re-draw plots."""
        self.densetc_plot.re_plot(axis_visible="on")
        self.densetc_plot.figure_canvas.draw()

    def _on_plot_flag_checkbox(self, checkbox, checked):
        """
        Shared handler for the five TC-display checkboxes. Each flips
        one boolean on the detail SitePlot and redraws; which boolean
        is looked up from the checkbox instance.

        TODO Fix mutually exclusive drawing flags
        """
        attr = self._plot_flag_checkboxes[checkbox]
        setattr(self.densetc_plot, attr, checked)
        if attr in ("use_lineplot", "use_heatmap"):
            self.bubble_slider.disabled = checked
            if checked:
                other = ("use_heatmap" if attr == "use_lineplot"
                         else "use_lineplot")
                setattr(self.densetc_plot, other, False)
        self.redraw()

    def on_mark_toggle(self, _event):
        """Event monitoring if site is 'marked' or not."""
        if self.mark_site_toggle.state == "down":
            if not self.densetc_plot.saved_marked:
                self.densetc_plot.on_changes_signal.send()
            self.densetc_plot.marked = True
        else:
            if self.densetc_plot.saved_marked:
                self.densetc_plot.on_changes_signal.send()
            self.densetc_plot.marked = False

    def on_reset(self, _event):
        """Event to reset any un-saved analysis changes made to a site."""
        self.unsaved_changes = False
        self.save_changes_button.disabled = True
        self.save_changes_button.background_color = [0.2, 0.65, 0, 1]
        self.reset_button.disabled = True
        self.reset_button.background_color = [0.25, 0.05, 0.1, 1]

        # Reset values to default
        self.densetc_plot.cf_idx = self.densetc_plot.saved_cf_idx
        self.densetc_plot.thresh_idx = self.densetc_plot.saved_thresh_idx
        self.densetc_plot.bw_idx = {
            lvl: v.copy() for lvl, v in self.densetc_plot.saved_bw_idx.items()}
        self.densetc_plot.continuous_bw_idx = \
            self.densetc_plot.saved_continuous_bw_idx.copy()
        self.densetc_plot.onset = self.densetc_plot.saved_onset
        self.densetc_plot.peak = self.densetc_plot.saved_peak
        self.densetc_plot.offset = self.densetc_plot.saved_offset
        self.densetc_plot.peak_driven_rate = \
            self.densetc_plot.saved_peak_driven_rate

        self.densetc_plot.contour_tc = \
            self.densetc_plot.saved_contour_tc.copy()

        self.redraw()

        # Reset Marked toggle, if necessary
        if self.densetc_plot.saved_marked:
            self.mark_site_toggle.state = "down"
        else:
            self.mark_site_toggle.state = "normal"

    def change_bin_size(self, _spinner, value):
        """Show PSTH with 1 or 5 ms bin size."""
        if value == "5 ms":
            self.densetc_plot.bin_size = 5
            self.densetc_plot.psth_plot()
        else:
            self.densetc_plot.bin_size = 1
            self.densetc_plot.psth_plot()
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
        """Update .json storage with new user analysis."""
        today = str(datetime.datetime.now())
        self.unsaved_changes = False
        self.save_changes_button.disabled = True
        self.save_changes_button.background_color = [0.2, 0.65, 0, 1]
        self.reset_button.disabled = True
        self.reset_button.background_color = [0.25, 0.05, 0.1, 1]
        frequencies = self.gui_instance.frequency
        intensities = self.gui_instance.intensity

        # Copy just in case, to prevent any dangling references
        continuous_bw = self.densetc_plot.continuous_bw_idx.copy()
        cf = self.densetc_plot.cf_idx
        thresh = self.densetc_plot.thresh_idx
        onset = self.densetc_plot.onset
        peak = self.densetc_plot.peak
        offset = self.densetc_plot.offset
        peak_driven_rate = self.densetc_plot.peak_driven_rate
        marked = self.densetc_plot.marked

        # Update 'saved' values to current values.
        self.densetc_plot.saved_cf_idx = cf
        self.densetc_plot.saved_thresh_idx = thresh
        bw = {lvl: v.copy() for lvl, v in self.densetc_plot.bw_idx.items()}
        self.densetc_plot.saved_bw_idx = {lvl: v.copy() for lvl, v in bw.items()}
        self.densetc_plot.saved_continuous_bw_idx = continuous_bw
        self.densetc_plot.saved_onset = onset
        self.densetc_plot.saved_peak = peak
        self.densetc_plot.saved_offset = offset
        self.densetc_plot.saved_peak_driven_rate = peak_driven_rate
        self.densetc_plot.saved_marked = marked

        # Finish analysis
        bw_khz, bw_oct = {}, {}
        for lvl in BW_LEVELS:
            if bw[lvl][0] is not None:
                bw_khz[lvl] = (frequencies[bw[lvl]] / 1000).tolist()
                bw_oct[lvl] = afunc.get_bandwidth(*frequencies[bw[lvl]]).tolist()
            else:
                bw_khz[lvl] = [None, None]
                bw_oct[lvl] = None

        if continuous_bw[0] is None:  
            # Site is being saved with new data, but cont. BW's haven't updated
            ttest_spike_counts = afunc.get_driven_vs_spont_spike_counts(
                self.densetc_plot.tuning_curve_df,
                driven_onset_ms=onset, 
                driven_offset_ms=offset,
                spont_onset_ms=400 - (offset - onset),
                spont_offset_ms=400)
            _, _, cf, thresh, *bws, continuous_bw, _ = \
                afunc.ttest_analyze_tuning_curve(
                     afunc.ttest_driven_vs_spont_tc(*ttest_spike_counts))
            bw = dict(zip(BW_LEVELS, bws))
        try:  
            # Cont. BW should work now, but rare cases may still create an 
            # exception (eg. no regions found in auto-tc)
            continuous_bw_khz = [(frequencies[bw] / 1000).tolist() for 
                                 bw in continuous_bw]
            continuous_bw_octave = [
                afunc.get_bandwidth(*frequencies[bw]).tolist() for 
                bw in continuous_bw]
        except TypeError:  
            # Cont. BW is likely [None, None] for some reason or other. 
            # In this case, leave it that way
            continuous_bw = [None, None]
            continuous_bw_khz = [None, None]
            continuous_bw_octave = None

        cf_khz = frequencies[cf] / 1000
        thresh_db = intensities[thresh].tolist()

        analysis_id = self.gui_instance.analysis_id
        site_number = self.map_number
        update_doc = {
            "cf_khz": cf_khz,
            "threshold_db": thresh_db,
            "cf_idx": cf,
            "threshold_idx": thresh,
            "continuous_bw_khz": continuous_bw_khz,
            "continuous_bw_idx": continuous_bw,
            "continuous_bw_octave": continuous_bw_octave,
            "onset_ms": onset,
            "peak_ms": peak,
            "offset_ms": offset,
            "peak_driven_rate_hz": peak_driven_rate,
            "marked": marked,
        }
        for lvl in BW_LEVELS:
            update_doc[f"bw{lvl}_idx"] = bw[lvl]
            update_doc[f"bw{lvl}_khz"] = bw_khz[lvl]
            update_doc[f"bw{lvl}_octave"] = bw_oct[lvl]

        self.gui_instance.densetc_analysis_collection.update_one(
            {"analysis_id": analysis_id, "number": site_number},
            {"$set": update_doc})

        self.gui_instance.analysis_metadata_collection.update_one(
            {"_id": analysis_id},
            {"$set": {
                "last_modified": today
            }})

        # Update plots with correct colors / values
        self.densetc_plot.bubble_color = self.densetc_plot.cf_cmap(
            self.densetc_plot.norm(self.densetc_plot.cf_idx))
        self.densetc_plot.lat_color = self.densetc_plot.speed_cmap(
            self.densetc_plot.speed_norm(self.densetc_plot.onset))
        self.redraw()

        self.gui_instance.plot_dict[self.map_number].cf_idx = cf
        self.gui_instance.plot_dict[self.map_number].thresh_idx = thresh
        self.gui_instance.plot_dict[self.map_number].onset = onset
        self.gui_instance.plot_dict[self.map_number].peak = peak
        self.gui_instance.plot_dict[self.map_number].offset = offset
        self.gui_instance.plot_dict[self.map_number].bw_idx = \
            {lvl: v.copy() for lvl, v in bw.items()}
        self.gui_instance.plot_dict[self.map_number].bubble_color = \
            self.densetc_plot.bubble_color
        self.gui_instance.plot_dict[self.map_number].lat_color = \
            self.densetc_plot.lat_color
        self.gui_instance.plot_dict[self.map_number].re_plot()

    def auto_tc_analyze(self, *_args, **_kwargs):
        """Run TC auto-analysis; use after manually updating PSTH latencies."""
        onset = self.densetc_plot.onset
        offset = self.densetc_plot.offset
        self.densetc_plot.on_changes_signal.send()
        ttest_spike_counts = afunc.get_driven_vs_spont_spike_counts(
            self.densetc_plot.tuning_curve_df,
            driven_onset_ms=onset, 
            driven_offset_ms=offset,
            spont_onset_ms=400 - (offset - onset),
            spont_offset_ms=400)
        smooth_tc, _, cf, thresh, *bws, continuous_bw, _ = \
            afunc.ttest_analyze_tuning_curve(
                afunc.ttest_driven_vs_spont_tc(*ttest_spike_counts))

        # Store analyzed data in the SitePlot object. 
        # Data is NOT saved until user hits 'Save' button
        self.densetc_plot.cf_idx = cf
        self.densetc_plot.thresh_idx = thresh
        self.densetc_plot.bw_idx = dict(zip(BW_LEVELS, bws))
        self.densetc_plot.continuous_bw_idx = continuous_bw
        smooth_tc[0 < smooth_tc] = 1
        self.densetc_plot.contour_tc = smooth_tc

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
        Updates latency lines and plots with any user analysis changes, and 
        marks whether a site is 'Marked' or has unsaved user changes.
        Triggers event to briefly flash Site in Map GUI to help user navigate
        where they were just inspecting.
        """
        xdata_onset = self.densetc_plot.onset_line.get_xdata()
        xdata_offset = self.densetc_plot.offset_line.get_xdata()
        self.gui_instance.plot_dict[
            self.map_number].onset_line.set_xdata(xdata_onset)
        self.gui_instance.plot_dict[
            self.map_number].offset_line.set_xdata(xdata_offset)
        self.gui_instance.plot_dict[
            self.map_number].onset = self.densetc_plot.onset
        self.gui_instance.plot_dict[
            self.map_number].offset = self.densetc_plot.offset
        self.gui_instance.plot_dict[self.map_number].update_bubble()
        try:
            self.gui_instance.plot_dict[self.map_number].figure_canvas.draw()
        except ValueError: # Raised by non-responsive sites -- just ignore. 
            pass
        
        self.flash_signal.send(self.map_number, 
                               unsaved_changes=self.unsaved_changes, 
                               marked=self.densetc_plot.marked)
        self.densetc_plot.active = False
        self.densetc_plot.fig.clf()
        self.manager.switch_to(self.gui_instance.parent)

    def on_pre_enter(self, *args):
        """Ready Site plots prior to switching GUI screens."""
        self.densetc_plot.active = True
        # Clear first plot generated during SitePlot.__init__()
        self.densetc_plot.fig.clf()
        self.densetc_plot.ax[0] = self.densetc_plot.fig.add_subplot(2, 1, 1)
        self.densetc_plot.ax[1] = self.densetc_plot.fig.add_subplot(2, 1, 2)
        self.densetc_plot.re_plot(axis_visible="on")
        self.densetc_plot.figure_canvas.draw()


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
        # Connect to signal for tracking transition from Site to Map Screens.
        self.flash_signal = blinker.signal("flash")
        self.flash_signal.connect(self.flash_cell)
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
        self.frequency = None
        self.intensity = None
        self.num_frequency = None
        self.num_intensity = None
        self.num_tones = None
        self.sites = None
        self.densetc_data = None
        self.densetc_analysis = None

        self.mongo_connection = None
        self.counter = 0
        self.site_screens = {}

        self.vor_df = None
        self.dense_df = None

        # Start with marks_active. Can be set to False before loading a map by
        # hitting the Show Fields button
        self.marks_active = True

        self.map_num = None
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
        self.vor_file = ""

        # Arrange GUI
        tools = StackLayout(orientation="lr-tb", size_hint=(0.075, 1))
        self.cf_spinner_label = Label(text="CF\n Colormap", 
                                      color=[0, 0, 0, 1], 
                                      size_hint=(1, 0.06), 
                                      halign="center")
        self.cf_colormap_dropdown = Spinner(
            text="viridis",
            size_hint=(1, 0.06),
            values={"viridis", "jet", "plasma", "inferno", "magma", "bone",
                    "cool", "tab20", "cubehelix", "gist_ncar"})
        self.heatmap_spinner_label = Label(text="Heatmap\n Colormap", 
                                           color=[0, 0, 0, 1], 
                                           size_hint=(1, 0.06),
                                           halign="center")
        self.heatmap_colormap_dropdown = Spinner(
            text="inferno", 
            size_hint=(1, 0.06), 
            values={"inferno", "viridis", "plasma", "magma", "ocean", 
                    "gnuplot2", "cubehelix", "jet", "bone", "gray"})
        self.cf_colormap_dropdown.bind(text=self.on_cf_colormap)
        self.heatmap_colormap_dropdown.bind(text=self.on_heatmap_colormap)

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
            self.mongo_connection = TinyMongoClient(
                os.path.dirname(self._db_path))
            self.subject_database = getattr(
                self.mongo_connection,
                os.path.splitext(os.path.basename(self._db_path))[0])
            self.map_metadata_collection = self.subject_database.metadata
            self.map_metadata = self.map_metadata_collection.find_one({})
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
            # configuration (the auto-analysis run). $exists is what we
            # want but tinymongo doesn't implement it; $ne against a
            # value the field never holds is a cheap substitute -- any
            # doc with the field matches, any doc without is skipped.
            self.project_configuration = \
                self.analysis_metadata_collection.find_one(
                    {"configuration": {"$ne": False}})["configuration"]
            self.frequency = np.sort(
                self.project_configuration["densetc_frequency_hz"])
            self.intensity = np.sort(
                self.project_configuration["densetc_intensity_db"])
            self.num_frequency = len(self.frequency)
            self.num_intensity = len(self.intensity)
            self.num_tones = \
                self.project_configuration["densetc_num_tones"]

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

            self.display_map()
            print("\n *** Ready! *** \n")

        except Exception as e:
            logging.exception(e)
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

        `self.parent` is the MapScreen that owns this layout.
        """
        self.parent.manager.switch_to(self.site_screens[site_number])

    def on_cf_colormap(self, _spinner, value):
        """Update bubble plot CF colormap using new selection."""
        for plot in self.plot_dict.values():
            plot.re_color(cf_cmap=value, 
                          heatmap_cmap=self.heatmap_colormap_dropdown.text)
            plot.figure_canvas.draw()
        for site in self.site_screens.values():
            # Do not draw, just update values for each Site
            site.densetc_plot.re_color(
                cf_cmap=value, 
                heatmap_cmap=self.heatmap_colormap_dropdown.text)

    def on_heatmap_colormap(self, _spinner, value):
        """Update spike heatmap colormap using new selection."""
        for plot in self.plot_dict.values():
            plot.re_color(cf_cmap=self.cf_colormap_dropdown.text, 
                          heatmap_cmap=value)
            plot.figure_canvas.draw()
        for site in self.site_screens.values():
            # Do not draw, just update values for each Site
            site.densetc_plot.re_color(cf_cmap=self.cf_colormap_dropdown.text,
                                       heatmap_cmap=value)

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
        attr = self._plot_flag_toggles[toggle]
        value = (toggle.state == "down")
        for plot in self.plot_dict.values():
            setattr(plot, attr, value)
        self._redraw_all_plots()

    def on_psth_ylim(self, _spinner, text):
        """Changing PSTH ylim's. Useful to emphasize weakly responsive sites."""
        self._redraw_all_plots(min_y=None if text == "None" else int(text))

    def export_map(self, _event):
        """Save Auditory Field selections and Marked sites to .json file."""
        if self.map_loaded:
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
        """Generate map visuals."""
        LineTuple = namedtuple("LineTuple", 
                               ["line", "color", "x_norm", "y_norm", 
                                "site_number"])
        MeshTuple = namedtuple("MeshTuple", 
                               ["mesh", "color", "x_norm", "y_norm", 
                                "site_number"])
        for site in self.sites:
            # xy coords are already normalized, but here we reduce them to 90%
            # to provide some padding at the border of MapLayout -> allows 
            # the user to move edge sites a little closer to the center for 
            # easier viewing. Purely aesthetic.
            reduced_scale = [0.1, 0.9]
            site_number = site["number"]
            site_analysis = self.densetc_analysis[site_number]
            # If 'marked' is not a current document property from database 
            # (analysis from older versions), add it.
            if "marked" not in site_analysis:
                site_analysis["marked"] = False

            # Recreate set of field and marked assignments saved in analysis 
            # for proper painting of sites
            if site_analysis["field_assignment"]:
                self.map_sets[
                    site_analysis["field_assignment"]].add(site_number)
            if site_analysis["marked"]:
                self.map_sets["Mark"].add(site_number)

            x = (site["voronoi_centroid"][0] * 
                 (reduced_scale[1] - reduced_scale[0]) / 
                 (1 - 0) + reduced_scale[0])
            y = (site["voronoi_centroid"][1] * 
                 (reduced_scale[1] - reduced_scale[0]) / 
                 (1 - 0) + reduced_scale[0])
            site_plot = SitePlot(
                size_hint=(None, None), 
                pos_hint={"center_x": x, "center_y": y},
                height=150, 
                width=200, 
                site_number=site_number,
                gui_instance=self,
                detailed_plot=False, 
                cf_cmap=self.cf_colormap_dropdown.text,
                heatmap_cmap=self.heatmap_colormap_dropdown.text)
            detail_plot = SitePlot(
                size_hint=(1, 1), 
                pos_hint={"center_x": 0.5, "center_y": 0.5},
                height=1, 
                width=2,
                site_number=site_number, 
                gui_instance=self,
                detailed_plot=True, 
                cf_cmap=self.cf_colormap_dropdown.text,
                heatmap_cmap=self.heatmap_colormap_dropdown.text)

            self.plot_dict[site_number] = site_plot
            self.map_canvas.add_widget(site_plot)
            self.site_screens[site_number] = SiteScreen(
                self, site_number, detail_plot, name=f"Site {site_number}")
            with self.map_canvas.canvas.before:
                # Check if site should start painted some color
                if site_analysis["field_assignment"] and not self.marks_active:
                    line_color = Color(*hex2rgb(
                        self.field_line_colors[
                            site_analysis["field_assignment"]]))
                    lw = 3
                elif site_analysis["marked"] and self.marks_active:
                    line_color = Color(*hex2rgb(
                        self.field_line_colors["Mark"]))
                    lw = 3
                else:
                    # Paint default color
                    line_color = Color(0.435, 0.51, 0.541, 1)  #xkcd:steel grey
                    lw = 1.5

                poly_norm_points = site["voronoi_vertices"]
                poly_x = [pnt[0] * (reduced_scale[1] - reduced_scale[0]) / 
                          (1 - 0) + reduced_scale[0] for pnt in 
                          poly_norm_points]
                poly_y = [pnt[1] * (reduced_scale[1] - reduced_scale[0]) / 
                          (1 - 0) + reduced_scale[0] for pnt in 
                          poly_norm_points]
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
                    (x, y, 0, 0) for x, y in 
                    zip(poly_x_adjusted, poly_y_adjusted)]))
                indices = list(range(len(poly_x_adjusted)))

                # Check if site should start painted some color
                # mesh_color must be declared AFTER line color is done being 
                # used, as Kivy uses a universal Color() rather than a keyword 
                # argument. Assigning line_color and mesh_color in the same 
                # if/else block would result in Lines and Meshes with 
                # identical colors.
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

        self.map_canvas.bind(size=self.update_line)
        self.map_canvas.bind(size=self.update_mesh)

        self.map_canvas.height = int(self.map_metadata["map_height"])
        self.map_canvas.width = int(self.map_metadata["map_width"])

        self.map_loaded = True

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
        # TODO tighten to `except KeyError` if verified same as `on_touch_up`
        try:
            if self.gui.paint_mode_active:
                touch.ud["line"].points += [touch.x, touch.y]
        except:
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
            # Error sometimes thrown when program tries to interpret a line
            # drawn over other GUI elements
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
    Render PSTH and TC matplotlib plots inside Kivy.
    Uses a flag to determine if obj is a MapScreen or SiteScreen (re: detailed)
    plot item.
    """
    def __init__(self, **kwargs):
        super(SitePlot, self).__init__(size_hint=kwargs["size_hint"],
                                       pos_hint=kwargs["pos_hint"],
                                       height=kwargs["height"], 
                                       width=kwargs["width"])

        # Allow detailed site plot and overview-map plot to use different 
        # settings by checking for a flag
        self.detailed_plot = kwargs["detailed_plot"]

        if self.detailed_plot:
            # Listen for signals
            self.on_changes_signal = blinker.Signal()
            self.on_cf_pick_signal = blinker.Signal()

        self.gui_instance = kwargs["gui_instance"]
        self.site_number = kwargs["site_number"]
        self.site_data = self.gui_instance.densetc_data[self.site_number]
        self.site_analysis = self.gui_instance.densetc_analysis[self.site_number]

        # Allow user to change cmaps used for plots
        self.cf_cmap = matplotlib.colormaps[kwargs["cf_cmap"]]
        self.heatmap_cmap = kwargs["heatmap_cmap"]
        # TODO test 48khz
        self.norm = matplotlib.colors.Normalize(
            vmin=0, vmax=self.gui_instance.num_frequency - 1)

        self.speed_cmap = cmocean.cm.speed
        if self.gui_instance.ic_bool:
            # 1ms-16ms with greater dynamic range for IC maps
            self.speed_norm = matplotlib.colors.PowerNorm(0.65, 
                                                          vmin=1, 
                                                          vmax=16)
        else:
            # 5ms-20ms with greater dynamic range
            self.speed_norm = matplotlib.colors.PowerNorm(0.65, 
                                                          vmin=5, 
                                                          vmax=20)

        # User changes will be reflected in non-saved variables. A site can be 
        # reset to the analysis default by copying the saved versions to the 
        # non-saved variables. Any changes the user wants to save will be 
        # copied into the saved variables, and then database will be updated.
        self.cf_idx = self.site_analysis["cf_idx"]
        self.saved_cf_idx = self.site_analysis["cf_idx"]
        self.thresh_idx = self.site_analysis["threshold_idx"]
        self.saved_thresh_idx = self.site_analysis["threshold_idx"]
        self.onset = self.site_analysis["onset_ms"]
        self.saved_onset = self.site_analysis["onset_ms"]
        self.offset = self.site_analysis["offset_ms"]
        self.saved_offset = self.site_analysis["offset_ms"]
        self.peak = self.site_analysis["peak_ms"]
        self.saved_peak = self.site_analysis["peak_ms"]
        self.peak_driven_rate = self.site_analysis["peak_driven_rate_hz"]
        self.saved_peak_driven_rate = self.site_analysis["peak_driven_rate_hz"]
        self.spont_rate = self.site_analysis["spont_firing_rate_hz"]
        self.bw_idx = {lvl: self.site_analysis[f"bw{lvl}_idx"].copy()
                       for lvl in BW_LEVELS}
        self.saved_bw_idx = {lvl: self.site_analysis[f"bw{lvl}_idx"].copy()
                             for lvl in BW_LEVELS}
        self.continuous_bw_idx = self.site_analysis["continuous_bw_idx"].copy()
        self.saved_continuous_bw_idx = self.site_analysis["continuous_bw_idx"].copy()
        try:
            self.sdf = self.site_analysis["bb_sdf"].copy()
        except KeyError:  # Analysis was made prior to versions adding sdf's
            self.sdf = 0
        try:
            self.saved_marked = self.site_analysis["marked"]
            self.marked = self.saved_marked
        except KeyError:  # Analysis was made prior to adding marks
            self.saved_marked = False
            self.marked = False

        # Initialize possible plot options
        self.manually_hidden = False
        self.bubble = None
        self.line = None
        self.heatmap = None
        self.psth = None
        self.fire_txt = None
        self.cf_txt = None
        self.latency_txt = None
        self.use_smooth_tc = False
        self.smooth_tuning_curve = None
        self.filtered_tuning_curve = None
        self.tuning_curve_contour = None
        self.use_lineplot = False
        self.use_heatmap = False
        self.cf_marker = None
        self.bw_lines = {lvl: None for lvl in BW_LEVELS}
        self.bw_markers = {lvl: [None, None] for lvl in BW_LEVELS}
        self.bw_press = {lvl: [False, False] for lvl in BW_LEVELS}
        self.contour_line = None
        self.picking_cf = False
        self.picking_bw = False
        self.bw_pressed = False
        self.use_bw = True
        if self.detailed_plot:
            self.active = False
            self.use_contour = False #True
            self.bin_size = 1
        else:
            self.active = True
            self.use_contour = False
            self.bin_size = 5

        self.sdf_line = None

        # Initialize latency/spont line properties
        self.onset_line = None
        self.offset_line = None
        self.spont_line = None

        # Initialize latency interaction flags (switched to 1 when user clicks 
        # on a latency line)
        self.onset_press = 0
        self.offset_press = 0

        # All bubbles in a plot will be drawn in relation to this maximum. 
        # Change if wanted.
        self.max_bubble_size = 6

        # Get TC for this site
        site_df = pd.DataFrame(self.site_data["spiketrains"])
        self.tuning_curve_df = afunc.get_tuning_curve_dataframe(site_df)
        self.raw_tuning_curve = np.array(self.tuning_curve_df.map(
                lambda x: 
                    afunc.remove_spont(x, 
                                       driven_onset_ms=self.onset, 
                                       driven_offset_ms=self.offset,
                                       spont_onset_ms=400 - (self.offset - 
                                                             self.onset),
                                       spont_offset_ms=400)
                    if ((x is not None) and (not np.any(np.isnan(x)))) 
                    else 0)).astype(np.uint8)

        self.ttest_spike_counts = afunc.get_driven_vs_spont_spike_counts(
            self.tuning_curve_df, 
            driven_onset_ms=self.onset, 
            driven_offset_ms=self.offset,
            spont_onset_ms=400 - (self.offset - self.onset),
            spont_offset_ms=400)
        self.ttest_tc = afunc.ttest_driven_vs_spont_tc(*self.ttest_spike_counts)
        self.saved_contour_tc = afunc.ttest_analyze_tuning_curve(self.ttest_tc)[0]
        # Threshold so only 1 contour level is drawn
        self.saved_contour_tc[0 < self.saved_contour_tc] = 1
        self.contour_tc = self.saved_contour_tc.copy()

        # Get values and indices from tuning curve array for bubble / line plot
        self.row, self.col = np.where(0 < self.raw_tuning_curve)
        self.val = self.raw_tuning_curve[self.row, self.col]

        # Generate bubble plot and psth
        self.fig, self.ax = plt.subplots(2, 1)
        self.ax[0].axis("off")
        self.ax[1].axis("off")
        self.raw_psth = np.array(self.site_analysis["psth"])
        if self.cf_idx is None:
            self.bubble_color = "r"
            self.lat_color = "m"
        else:
            self.bubble_color = self.cf_cmap(self.norm(self.cf_idx))
            self.lat_color = self.speed_cmap(self.speed_norm(self.onset))

        self.bubble_plot()
        self.psth_plot()

        # Aesthetics
        self.fig.patch.set_alpha(0)
        self.fig.subplots_adjust(wspace=0, hspace=0)
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

    def mouse_click_event(self, event):
        """User interaction with latency or bandwidth lines."""
        event.x, event.y = self.to_window(*self.to_parent(event.x, event.y))
        if self.detailed_plot:
            if self.active:
                """
                Checking event.inaxes is unpredictable when x, y need 
                transformation (as they do here), since event.inaxes is created
                before the transformation is done (and will return None in some
                cases at the edge of axes. No good!). Instead, transform x, y 
                into axes to check if point falls inside bounding box (values 
                between 0-1 fall inside axes, other values are outside).
                """
                x_coor_ax0, y_coor_ax0 = \
                    self.ax[0].transAxes.inverted().transform([event.x, 
                                                               event.y])
                x_coor_ax1, y_coor_ax1 = \
                    self.ax[1].transAxes.inverted().transform([event.x, 
                                                               event.y])
                if (0 <= x_coor_ax0 <= 1) and (0 <= y_coor_ax0 <= 1):
                    self.on_pick_line(event)
                elif (0 <= x_coor_ax1 <= 1) and (0 <= y_coor_ax1 <= 1):
                    if self.picking_cf:
                        self.pick_cf(event)
                    elif self.use_bw:
                        # Only allow pick_bw() if bw's are visible
                        self.pick_bw(event)
        else:
            # Map-wide axes coords don't line up with event coords even after
            # this transformation
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

    def re_color(self, cf_cmap="viridis", heatmap_cmap="inferno"):
        """Update bubble plot or heatmap colors."""
        self.heatmap_cmap = heatmap_cmap
        self.cf_cmap = matplotlib.colormaps[cf_cmap]
        if self.cf_idx is not None:
            # TODO allow user to change No CF color (default is red)
            self.bubble_color = self.cf_cmap(self.norm(self.cf_idx))
        if self.use_heatmap:
            self.heatmap.set_cmap(self.heatmap_cmap)
        elif not self.use_lineplot:
            self.bubble.update({"facecolors": self.bubble_color})

    def re_plot(self, axis_visible="off", min_y=None):
        """Re-plot TC."""
        if self.use_lineplot:
            self.line_plot(axis_visible=axis_visible)
        elif self.use_heatmap:
            self.heatmap_plot(axis_visible=axis_visible)
        else:
            self.bubble_plot(axis_visible=axis_visible)
        self.psth_plot(min_y=min_y)

    def on_pick_line(self, event):
        """Initial UX for user updating latency line."""
        if self.active:
            if self.detailed_plot:
                lat_lw = 5
            else:
                lat_lw = 1.5
            if self.onset_line.contains(event)[0]:
                self.onset_line.set_lw(lat_lw)
                self.onset_press = 1
                event.canvas.draw()

            elif self.offset_line.contains(event)[0]:
                self.offset_line.set_lw(lat_lw)
                self.offset_press = 1
                event.canvas.draw()

    def move_line(self, x, y):
        """Ongoing UX for user updating latency line."""
        # trans_x/y are in display coordinates (see MPL doc for info for
        # definition)
        # Must transform into axes user data coordinates (xlim, ylim) in order
        # to move line to appropriate x-coordinate based on mouse position
        ax_inv = self.ax[0].transData.inverted()
        xdata, ydata = ax_inv.transform((x, y))
        if xdata is None:
            return

        # TODO generalize sweep length
        if self.onset_press and (0 <= xdata <= 400):
            if xdata < self.offset_line.get_xdata()[0]:
                self.onset_line.set_xdata([xdata, xdata])
                self.onset = int(round(xdata))
                if self.detailed_plot:
                    self.on_changes_signal.send()
                    peak_hist = self.raw_psth.copy()
                    peak_hist[0:self.onset] = peak_hist[self.offset:] = 0
                    self.peak = int(np.argmax(peak_hist))

                    self.peak_driven_rate = \
                        afunc.get_peak_driven_rate(
                            self.raw_psth[self.onset:self.offset],
                            self.spont_rate,
                            self.gui_instance.num_tones)
                    self.latency_txt.set_text(
                        f"{self.onset}, {self.peak}, {self.offset}")

                if self.use_lineplot:
                    self.update_line()
                elif self.use_heatmap:
                    self.update_heatmap()
                else:
                    self.update_bubble()
                self.figure_canvas.draw()

        # TODO generalize sweep length
        elif self.offset_press and (0 <= xdata <= 400):
            if xdata > self.onset_line.get_xdata()[0]:
                self.offset_line.set_xdata([xdata, xdata])
                self.offset = int(round(xdata))
                if self.detailed_plot:
                    self.on_changes_signal.send()
                    peak_hist = self.raw_psth.copy()
                    peak_hist[0:self.onset] = peak_hist[self.offset:] = 0
                    self.peak = int(np.argmax(peak_hist))

                    self.peak_driven_rate = \
                        afunc.get_peak_driven_rate(
                            self.raw_psth[self.onset:self.offset],
                            self.spont_rate,
                            self.gui_instance.num_tones)
                    self.latency_txt.set_text(
                        f"{self.onset}, {self.peak}, {self.offset}")
                    
                if self.use_lineplot:
                    self.update_line()
                elif self.use_heatmap:
                    self.update_heatmap()
                else:
                    self.update_bubble()
                self.figure_canvas.draw()

    def update_heatmap(self):
        """Update heatmap without fully re-plotting."""
        if self.use_smooth_tc:
            ttest_spike_counts = afunc.get_driven_vs_spont_spike_counts(
                self.tuning_curve_df, 
                driven_onset_ms=self.onset, 
                driven_offset_ms=self.offset,
                spont_onset_ms=400 - (self.offset - self.onset),
                spont_offset_ms=400)
            tc_image = afunc.ttest_driven_vs_spont_tc(*ttest_spike_counts)
        else:
            self.raw_tuning_curve = np.array(self.tuning_curve_df.map(
                lambda x: 
                    afunc.remove_spont(x, 
                                       driven_onset_ms=self.onset, 
                                       driven_offset_ms=self.offset,
                                       spont_onset_ms=400 - (self.offset - 
                                                             self.onset),
                                       spont_offset_ms=400)
                    if ((x is not None) and (not np.any(np.isnan(x))))
                    else 0)).astype(np.uint8)
            tc_image = self.raw_tuning_curve

        self.heatmap.set_data(tc_image)

    def update_line(self):
        """Update lineplot without fully re-plotting."""
        max_line_length = 1
        self.raw_tuning_curve = np.array(self.tuning_curve_df.map(
            lambda x: 
                afunc.remove_spont(x, 
                                   driven_onset_ms=self.onset, 
                                   driven_offset_ms=self.offset,
                                   spont_onset_ms=400 - (self.offset - 
                                                         self.onset),
                                   spont_offset_ms=400) 
                if x is not None else 0)).astype(np.uint8)
        if self.use_smooth_tc:
            self.filtered_tuning_curve = afunc.analyze_tuning_curve(
                self.raw_tuning_curve)[1]
            self.row, self.col = np.where(self.filtered_tuning_curve > 0)
            self.val = self.filtered_tuning_curve[self.row, self.col]
        else:
            self.row, self.col = np.where(self.raw_tuning_curve > 0)
            self.val = self.raw_tuning_curve[self.row, self.col]

        x = self.col
        y = self.row
        s = self.val

        try:
            scaled_s = minmax_scale(list(s) + [0], 
                                    feature_range=(0, max_line_length))[:-1]
        except TypeError:
            # Thrown if s contains no values (non-responsive site)
            scaled_s = s

        line_list = [[[x_, y_ + 0.25], [x_, y_ + 0.25 - s_]] for 
                     x_, y_, s_ in zip(x, y, scaled_s)]
        self.line.set_segments(line_list)

    def update_bubble(self):
        """Update bubble plot without fully re-plotting."""
        if self.use_smooth_tc:
            ttest_spike_counts = afunc.get_driven_vs_spont_spike_counts(
                self.tuning_curve_df, 
                driven_onset_ms=self.onset, 
                driven_offset_ms=self.offset,
                spont_onset_ms=400 - (self.offset - self.onset),
                spont_offset_ms=400)
            self.filtered_tuning_curve = \
                afunc.ttest_driven_vs_spont_tc(*ttest_spike_counts)
            self.row, self.col = np.where(0 < self.filtered_tuning_curve)
            self.val = self.filtered_tuning_curve[self.row, self.col]
        else:
            self.raw_tuning_curve = np.array(self.tuning_curve_df.map(
                lambda x: 
                    afunc.remove_spont(x, 
                                       driven_onset_ms=self.onset, 
                                       driven_offset_ms=self.offset,
                                       spont_onset_ms=400 - (self.offset - 
                                                             self.onset),
                                       spont_offset_ms=400)
                    if ((x is not None) and (not np.any(np.isnan(x))))
                    else 0)).astype(np.uint8)
            self.row, self.col = np.where(self.raw_tuning_curve > 0)
            self.val = self.raw_tuning_curve[self.row, self.col]

        x = self.col
        y = self.row
        s = self.val

        """
        Scale bubble size against a maximum value. Add [0] to ensure entire 
        dynamic range is used (otherwise lowest spike value will default to 
        lowest bubble size -- here, 0).
        """
        try:
            scaled_s = minmax_scale(
                list(s) + [0], feature_range=(0, self.max_bubble_size))[:-1]
        except TypeError:
            # Thrown if s contains no values (non-responsive site)
            scaled_s = s

        offsets = np.column_stack((x, y))
        self.bubble.update({"offsets": offsets, "sizes": scaled_s ** 2})

    def update_bubble_size(self):
        """
        Quick function to only update the size of the bubbles. Call 
        update_bubble() instead if positions or values need updating.
        """
        s = self.val

        """
        Scale bubble size against a maximum value. Add [0] to ensure entire 
        dynamic range is used (otherwise lowest spike value will default to 
        lowest bubble size -- here, 0).
        """
        try:
            scaled_s = minmax_scale(
                list(s) + [0], feature_range=(0, self.max_bubble_size))[:-1]
        except TypeError:
            # Thrown if s contains no values (non-responsive site)
            scaled_s = s

        self.bubble.update({"sizes": scaled_s ** 2})

    def _draw_tc_overlays(self, ax, contour_color=None):
        """
        Draw CF marker, BW lines/markers, and contour on top of whichever
        TC rendering (bubble / line / heatmap) just populated `ax`.

        `contour_color` lets the dark-background plots (line, heatmap in
        detailed view) force a white contour; None uses matplotlib's cycle.
        """
        if self.use_bw and (self.cf_idx is not None):
            for lvl in BW_LEVELS:
                idx = self.bw_idx[lvl]
                if idx[0] is None:
                    continue
                y = self.thresh_idx + lvl // 5  # assumes 5 dB steps — TODO issue #13
                self.bw_lines[lvl] = ax.plot(idx, [y, y], "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw_markers[lvl][0] = ax.plot(
                        idx[0], y, "rd", ms=8, picker=5)[0]
                    self.bw_markers[lvl][1] = ax.plot(
                        idx[1], y, "rd", ms=8, picker=5)[0]

        if self.use_contour:
            if contour_color:
                self.contour_line = ax.contour(self.contour_tc, levels=[0],
                                               colors=contour_color)
            else:
                self.contour_line = ax.contour(self.contour_tc, levels=[0])

        if self.cf_idx is not None:
            self.cf_marker = ax.plot(self.cf_idx, self.thresh_idx,
                                     "r*", ms=8, alpha=0.5)[0]

    def bubble_plot(self, ax=None, x=None, y=None, s=None, color=None, 
                    axis_visible="off", axis_color="xkcd:white"):
        """
        Done for SiteScreen.__init__(). It updates bubble size and axis, but 
        doesn't (currently) have this data so it was easier at the time of 
        writing to just make the defaults available and simply call 
        .bubble_plot() to get modified version of an existing plot.
        """
        if ax is None:
            ax = self.ax[1]
        if None in [x, y, s]:
            # User must pass all 3 kwargs if they want to plot something 
            # different than default plot behavior

            if self.use_smooth_tc:
                ttest_spike_counts = afunc.get_driven_vs_spont_spike_counts(
                    self.tuning_curve_df, 
                    driven_onset_ms=self.onset, 
                    driven_offset_ms=self.offset,
                    spont_onset_ms=400 - (self.offset - self.onset),
                    spont_offset_ms=400)
                self.filtered_tuning_curve = \
                    afunc.ttest_driven_vs_spont_tc(*ttest_spike_counts)
                self.row, self.col = np.where(0 < self.filtered_tuning_curve)
                self.val = self.filtered_tuning_curve[self.row, self.col]
            else:
                self.raw_tuning_curve = np.array(self.tuning_curve_df.map(
                    lambda x: 
                        afunc.remove_spont(x, 
                                           driven_onset_ms=self.onset,
                                           driven_offset_ms=self.offset,
                                           spont_onset_ms=400 - (self.offset -
                                                                 self.onset),
                                           spont_offset_ms=400)
                       if ((x is not None) and (not np.any(np.isnan(x))))
                       else 0)).astype(np.uint8)
                self.row, self.col = np.where(0 < self.raw_tuning_curve)
                self.val = self.raw_tuning_curve[self.row, self.col]

            x = self.col
            y = self.row
            s = self.val

        if color is None:
            color = self.bubble_color

        ax.clear()
        """
        Scale bubble size against a maximum value. Add [0] to ensure entire 
        dynamic range is used (otherwise lowest spike value will default to 
        lowest bubble size -- here, 0).
        """
        try:
            scaled_s = minmax_scale(
                list(s) + [0], feature_range=(0, self.max_bubble_size))[:-1]
        except TypeError:
            # Thrown if s contains no values (non-responsive site)
            scaled_s = s
        self.bubble = ax.scatter(x=x, y=y, s=scaled_s ** 2, edgecolors="black",
                                 lw=0.5, color=color)
        ax.set_facecolor(axis_color)

        self._draw_tc_overlays(ax)

        ax.set_xlim([0, self.gui_instance.num_frequency])
        ax.set_ylim([0, self.gui_instance.num_intensity])
        ax.axis(axis_visible)

    def line_plot(self, ax=None, x=None, y=None, s=None, axis_visible="on", 
                  axis_color="xkcd:black"):
        """
        Old tc_explore style line plot.
        """
        max_line_length = 1
        if ax is None:
            ax = self.ax[1]
        if None in [x, y, s]:
            # User must pass all 3 kwargs if they want to plot something 
            # different than default plot behavior
            self.raw_tuning_curve = np.array(self.tuning_curve_df.map(
                lambda x: 
                    afunc.remove_spont(x, 
                                       driven_onset_ms=self.onset, 
                                       driven_offset_ms=self.offset,
                                       spont_onset_ms=400 - (self.offset - 
                                                             self.onset),
                                       spont_offset_ms=400)
                    if ((x is not None) and (not np.any(np.isnan(x))))
                    else 0)).astype(np.uint8)
            if self.use_smooth_tc:
                self.filtered_tuning_curve = afunc.analyze_tuning_curve(
                    self.raw_tuning_curve)[1]
                self.row, self.col = np.where(self.filtered_tuning_curve > 0)
                self.val = self.filtered_tuning_curve[self.row, self.col]
            else:
                self.row, self.col = np.where(self.raw_tuning_curve > 0)
                self.val = self.raw_tuning_curve[self.row, self.col]

            x = self.col
            y = self.row
            s = self.val

        ax.clear()
        try:
            scaled_s = minmax_scale(list(s) + [0], 
                                    feature_range=(0, max_line_length))[:-1]
        except TypeError:
            # Thrown if s contains no values (non-responsive site)
            scaled_s = s

        line_list = [[[x_, y_+0.25], [x_, y_+0.25 - s_]] for 
                     x_, y_, s_ in zip(x, y, scaled_s)]
        self.line = LineCollection(line_list, linewidths=2, colors="y")
        ax.add_collection(self.line)
        ax.set_facecolor(axis_color)

        self._draw_tc_overlays(
            ax, contour_color="xkcd:white" if self.detailed_plot else None)
        
        ax.set_xlim([0, self.gui_instance.num_frequency])
        ax.set_ylim([0, self.gui_instance.num_intensity])
        ax.axis(axis_visible)

    def heatmap_plot(self, ax=None, tc_image=None, axis_visible="on"):
        """
        I like heatmaps and bubbles.
        """
        if ax is None:
            ax = self.ax[1]
        if tc_image is None:
            # User must pass all 3 kwargs if they want to plot something 
            # different than default plot behavior
            self.raw_tuning_curve = np.array(self.tuning_curve_df.map(
                lambda x: 
                    afunc.remove_spont(x, 
                                       driven_onset_ms=self.onset, 
                                       driven_offset_ms=self.offset,
                                       spont_onset_ms=400 - (self.offset - 
                                                             self.onset),
                                       spont_offset_ms=400)
                    if ((x is not None) and (not np.any(np.isnan(x))))
                    else 0)).astype(np.uint8)
            if self.use_smooth_tc:
                tc_image = afunc.analyze_tuning_curve(self.raw_tuning_curve)[1]
            else:
                tc_image = self.raw_tuning_curve

        ax.clear()
        self.heatmap = ax.imshow(tc_image, cmap=self.heatmap_cmap, 
                                 aspect="auto")

        self._draw_tc_overlays(
            ax, contour_color="xkcd:white" if self.detailed_plot else None)
        
        ax.set_xlim([0, self.gui_instance.num_frequency-1])
        ax.set_ylim([0, self.gui_instance.num_intensity-1])
        ax.axis(axis_visible)

    def psth_plot(self, ax=None, axis_visible="off", bin_size=None, 
                  sweep_length=399, min_y=None):
        """
        Plots PSTH (originally done in __init__; breaking it out allows updates 
        to bin size, colors, markers, spont line etc.
        """
        if ax is None:
            ax = self.ax[0]
        if bin_size is None:
            bin_size = self.bin_size

        ax.clear()
        if bin_size in [1, 5]:  # Currently only 1 and 5ms are supported.
            # Assumes 400ms tone sweep. Pass value for speech, or other.
            num_bins = round(sweep_length / bin_size)
            psth_binned = np.histogram(range(len(self.raw_psth)), 
                                       bins=num_bins, weights=self.raw_psth)[0]
        else:
            # raw_psth should already be 1ms binned.
            psth_binned = self.raw_psth
            num_bins = len(psth_binned)

        hist_peak = np.argmax(psth_binned)
        """
        Get peak spike rate. This is not the same as driven rate. It also 
        changes depending on bin-size selection. This is just to be used as a 
        quick visual tool for inspecting site without needing psth y-axis 
        clutter.
        """
        if bin_size == 5:
            ms_multiplier = 200  # Get rate in Hz; 5ms * 200 = 1s
        else:
            ms_multiplier = 1000  # 1ms * 1000
        peak_spike_rate = int(round((psth_binned[hist_peak] * ms_multiplier) / 
                                    self.gui_instance.num_tones))

        # Plot psth with text showing peak-firing rate and onset, peak, offset
        # latencies
        self.psth = ax.hist(range(len(self.raw_psth)), weights=self.raw_psth, 
                            bins=num_bins, alpha=1, color=self.lat_color, 
                            edgecolor="#fdfdfe", lw=0.4, histtype="stepfilled")

        if self.detailed_plot:
            self.sdf_line = ax.plot(np.array(self.sdf)*bin_size*self.gui_instance.num_tones, lw=2, 
                                    color="xkcd:amber")[0]

        # If a minimum max y-lim value is set (so small Hz do indeed look 
        # small), set it IF it is larger than current
        if min_y:
            # Convert min_y (in Hz) to # of spikes (ylim value)
            min_y = (min_y / ms_multiplier) * self.gui_instance.num_tones
            ylim = ax.get_ylim()
            if ylim[1] < min_y:
                ax.set_ylim([0, min_y])
                y_val = min_y
            else:
                y_val = psth_binned[hist_peak]
        else:
            y_val = psth_binned[hist_peak]

        if self.cf_idx is None:
            cf_val = "-"
        else:
            cf_val = f"{self.gui_instance.frequency[self.cf_idx]/1000:.1f}"

        self.latency_txt = ax.annotate(
            f"On: {self.onset}, Pk: {self.peak}, Off: {self.offset}\n"
            f"Rate: {peak_spike_rate} Hz, CF: {cf_val} kHz",
            (1.25, 0), 
            xytext=[self.offset+5, y_val], 
            size=10, 
            va="top", 
            name="Segoe UI",
            weight="bold", 
            color="xkcd:dark blue")

        ax.set_xlim([0, sweep_length-1])

        # If detailed plot, plot spontaneous and SDF
        if self.detailed_plot:
            # Spont was calculated at 1ms bin size
            spont = (self.spont_rate / 1000) * bin_size * self.gui_instance.num_tones
            self.spont_line = ax.plot([0, sweep_length-1], [spont, spont], 
                                      "c", lw=2)[0]

        # Plot latency lines on psth
        if self.detailed_plot:
            lat_lw = 3
        else:
            lat_lw = 1

        self.onset_line = ax.plot([self.onset, self.onset], [0, y_val],
                                  "r", lw=lat_lw, picker=2)[0]
        self.offset_line = ax.plot([self.offset, self.offset], [0, y_val],
                                   "r", lw=lat_lw, picker=2)[0]

        ax.axis(axis_visible)

    def off_pick(self):
        """Final UX for user updating latency line."""
        if self.detailed_plot:
            lat_lw = 3
        else:
            lat_lw = 1
        self.onset_line.set_lw(lat_lw)
        self.onset_press = 0
        self.offset_line.set_lw(lat_lw)
        self.offset_press = 0
        if self.detailed_plot:
            # TODO probably can condense with above, but just testing right now
            # Update psth with new firing rate / text positions
            self.psth_plot()

        self.figure_canvas.draw()

    def pick_cf(self, event):
        """
        Quick function to let user manually select new cf and threshold from 
        tuning curve plot based on mouse_click_event..
        Must do this roundabout way of checking, because ginput() 
        implementation appears to crash Kivy
        """
        # Transform Kivy event coords into figure coords, then check if mouse
        # event occurs inside axes coords (also transformed into figure 
        # coords). If event is outside of x or y limits of TC axis, 
        # return (ignore)
        xdata, ydata = self.ax[1].transData.inverted().transform(
            (event.x, event.y))
        if (xdata is None) or (ydata is None):
            return
        # Update CF and Thresh, and move CF marker to new position
        self.cf_idx = int(round(xdata))
        self.thresh_idx = int(round(ydata))
        if self.cf_marker is None:
            self.cf_marker = self.ax[1].plot(self.cf_idx, self.thresh_idx, 
                                             "r*", ms=8, alpha=0.5)[0]
        else:
            self.cf_marker.set_xdata([self.cf_idx])
            self.cf_marker.set_ydata([self.thresh_idx])
        freq_range = self.gui_instance.num_frequency
        int_range = self.gui_instance.num_intensity
        # Any BW whose row now sits above the intensity grid is cleared;
        # any that's newly in-range but was previously absent gets a wide
        # default so the user has handles to drag.
        for lvl in BW_LEVELS:
            row = self.thresh_idx + lvl // 5  # assumes 5 dB steps — TODO issue #13
            if row <= int_range:
                if self.bw_idx[lvl][0] is None:
                    self.bw_idx[lvl] = [10, freq_range - 10]
            else:
                self.bw_idx[lvl] = [None, None]

        # Un-flag picking_cf
        self.picking_cf = False
        # Signal that cf was picked
        self.on_cf_pick_signal.send()
        self.on_changes_signal.send()

    def pick_bw(self, event):
        """
        Bubbles and line plots update y-axis position of bw10-40. Picking a 
        marker on the end of these lines will allow the user to adjust the 
        x-axis position of bw's at a site. Similar to pick and move of latency
        lines.
        """
        # TODO currently doesn't do anything to continuous_bw
        for lvl in BW_LEVELS:
            markers = self.bw_markers[lvl]
            if markers[0] is None:
                continue
            for side in (0, 1):
                if markers[side].contains(event)[0]:
                    self.bw_pressed = True
                    markers[side].set_ms(12)
                    self.bw_press[lvl][side] = True
                    event.canvas.draw()
                    return  # one marker at a time

    def move_bw(self, event_x, event_y):
        """
        Drag the currently-held BW marker, clamped to [0, max_freq_idx] and
        prevented from crossing its partner.
        """
        ax_inv = self.ax[1].transData.inverted()
        xdata, _ = ax_inv.transform((event_x, event_y))
        if xdata is None:
            return
        max_idx = self.gui_instance.num_frequency - 1

        for lvl in BW_LEVELS:
            press = self.bw_press[lvl]
            markers = self.bw_markers[lvl]
            idx = self.bw_idx[lvl]
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
        """Final UX for user updating bandwidth."""
        self.bw_pressed = False
        for lvl in BW_LEVELS:
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
