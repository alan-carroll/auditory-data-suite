import datetime
import itertools
import os
import numpy as np
import matplotlib
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
import matplotlib.pyplot as plt
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.clock import Clock
from kivy.graphics import Color, Line, Mesh
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pandas as pd
import cmocean
from kivy.utils import get_color_from_hex as hex2rgb
from kivy.uix.screenmanager import Screen, ScreenManager
import tkinter as tk
from tkinter import messagebox
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
    def __init__(self):
        """
        Uses `ScreenManager` to allow swapping between main Map GUI and
        Site-specific GUIs.
        """
        super(FieldSelectionApp, self).__init__()
        self.SM = ScreenManager()
        self.map_screen = MapScreen(name="Map")
        self.SM.add_widget(self.map_screen)
        self.SM.current = "Map"

    def build(self):
        return self.SM


class MapScreen(Screen):
    def __init__(self, **kwargs):
        """Container for main application."""
        super(MapScreen, self).__init__(**kwargs)
        self.add_widget(FieldSelectionGUI())


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
        self.contour_checkbox.bind(active=self.on_contour_checkbox)
        self.contour_checkbox_label = Label(text="TC Contour: ", 
                                            color=[0, 0, 0, 1], 
                                            size_hint=(1, 0.5))
        contour_layout.add_widget(self.contour_checkbox_label)
        contour_layout.add_widget(self.contour_checkbox)

        smooth_tc_layout = BoxLayout(orientation="vertical", 
                                     size_hint=(0.1, 1))
        self.smooth_tc_checkbox = CheckBox(active=False, size_hint=(1, 0.5))
        self.smooth_tc_checkbox.bind(active=self.on_smooth_tc_checkbox)
        self.smooth_tc_checkbox_label = Label(text="Smooth TC: ", 
                                              color=[0, 0, 0, 1], 
                                              size_hint=(1, 0.5))
        smooth_tc_layout.add_widget(self.smooth_tc_checkbox_label)
        smooth_tc_layout.add_widget(self.smooth_tc_checkbox)

        lineplot_layout = BoxLayout(orientation="vertical", size_hint=(0.1, 1))
        self.lineplot_checkbox = CheckBox(active=False, size_hint=(1, 0.5))
        self.lineplot_checkbox.bind(active=self.on_lineplot_checkbox)
        self.lineplot_checkbox_label = Label(text="TC Line Plot: ", 
                                             color=[0, 0, 0, 1], 
                                             size_hint=(1, 0.5))
        lineplot_layout.add_widget(self.lineplot_checkbox_label)
        lineplot_layout.add_widget(self.lineplot_checkbox)

        heatmap_layout = BoxLayout(orientation="vertical", size_hint=(0.1, 1))
        self.heatmap_checkbox = CheckBox(active=False, size_hint=(1, 0.5))
        self.heatmap_checkbox.bind(active=self.on_heatmap_checkbox)
        self.heatmap_checkbox_label = Label(text="TC Heatmap: ", 
                                            color=[0, 0, 0, 1], 
                                            size_hint=(1, 0.5))
        heatmap_layout.add_widget(self.heatmap_checkbox_label)
        heatmap_layout.add_widget(self.heatmap_checkbox)

        bw_layout = BoxLayout(orientation="vertical", size_hint=(0.1, 1))
        self.bw_checkbox = CheckBox(active=True, size_hint=(1, 0.5))
        self.bw_checkbox.bind(active=self.on_bw_checkbox)
        self.bw_checkbox_label = Label(text="Show BWs: ", color=[0, 0, 0, 1], 
                                       size_hint=(1, 0.5))
        bw_layout.add_widget(self.bw_checkbox_label)
        bw_layout.add_widget(self.bw_checkbox)

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
        self.densetc_plot.bw10_idx = self.densetc_plot.saved_bw10_idx.copy()
        self.densetc_plot.bw20_idx = self.densetc_plot.saved_bw20_idx.copy()
        self.densetc_plot.bw30_idx = self.densetc_plot.saved_bw30_idx.copy()
        self.densetc_plot.bw40_idx = self.densetc_plot.saved_bw40_idx.copy()
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

    def on_contour_checkbox(self, _checkbox, checked):
        """Display smoothed contour lines around TC."""
        if checked:
            self.densetc_plot.use_contour = True
        else:
            self.densetc_plot.use_contour = False
        self.redraw()

    def on_smooth_tc_checkbox(self, _checkbox, checked):
        """Display smoothed TC instead of raw spike counts."""
        if checked:
            self.densetc_plot.use_smooth_tc = True
        else:
            self.densetc_plot.use_smooth_tc = False
        self.redraw()

    def on_lineplot_checkbox(self, _checkbox, checked):
        """Display spike counts as lines. Longer -> more spikes."""
        if checked:
            self.densetc_plot.use_lineplot = True
            self.bubble_slider.disabled = True
            self.densetc_plot.use_heatmap = False
        else:
            self.densetc_plot.use_lineplot = False
            self.bubble_slider.disabled = False
        self.redraw()

    def on_heatmap_checkbox(self, _checkbox, checked):
        """Display spike counts as a heatmap."""
        if checked:
            self.densetc_plot.use_heatmap = True
            self.bubble_slider.disabled = True
            self.densetc_plot.use_lineplot = False
        else:
            self.densetc_plot.use_heatmap = False
            self.bubble_slider.disabled = False
        self.redraw()

    def on_bw_checkbox(self, _checkbox, checked):
        """Display 10-40 dB Bandwidths on top of TC."""
        if checked:
            self.densetc_plot.use_bw = True
        else:
            self.densetc_plot.use_bw = False
        self.redraw()

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
        bw10 = self.densetc_plot.bw10_idx.copy()
        bw20 = self.densetc_plot.bw20_idx.copy()
        bw30 = self.densetc_plot.bw30_idx.copy()
        bw40 = self.densetc_plot.bw40_idx.copy()
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
        self.densetc_plot.saved_bw10_idx = bw10
        self.densetc_plot.saved_bw20_idx = bw20
        self.densetc_plot.saved_bw30_idx = bw30
        self.densetc_plot.saved_bw40_idx = bw40
        self.densetc_plot.saved_continuous_bw_idx = continuous_bw
        self.densetc_plot.saved_onset = onset
        self.densetc_plot.saved_peak = peak
        self.densetc_plot.saved_offset = offset
        self.densetc_plot.saved_peak_driven_rate = peak_driven_rate
        self.densetc_plot.saved_marked = marked

        # Finish analysis
        if bw10[0] is not None:
            bw10_khz = (frequencies[bw10] / 1000).tolist()
            bw10_octave = afunc.get_bandwidth(*frequencies[bw10]).tolist()
        else:
            bw10_khz = [None, None]
            bw10_octave = None
        if bw20[0] is not None:
            bw20_khz = (frequencies[bw20] / 1000).tolist()
            bw20_octave = afunc.get_bandwidth(*frequencies[bw20]).tolist()
        else:
            bw20_khz = [None, None]
            bw20_octave = None
        if bw30[0] is not None:
            bw30_khz = (frequencies[bw30] / 1000).tolist()
            bw30_octave = afunc.get_bandwidth(*frequencies[bw30]).tolist()
        else:
            bw30_khz = [None, None]
            bw30_octave = None
        if bw40[0] is not None:
            bw40_khz = (frequencies[bw40] / 1000).tolist()
            bw40_octave = afunc.get_bandwidth(*frequencies[bw40]).tolist()
        else:
            bw40_khz = [None, None]
            bw40_octave = None

        if continuous_bw[0] is None:  
            # Site is being saved with new data, but cont. BW's haven't updated
            ttest_spike_counts = afunc.get_driven_vs_spont_spike_counts(
                self.densetc_plot.tuning_curve_df,
                driven_onset_ms=onset, 
                driven_offset_ms=offset,
                spont_onset_ms=400 - (offset - onset),
                spont_offset_ms=400)
            _, _, cf, thresh, bw10, bw20, bw30, bw40, continuous_bw, _ = \
                afunc.ttest_analyze_tuning_curve(
                     afunc.ttest_driven_vs_spont_tc(*ttest_spike_counts))
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
        self.gui_instance.densetc_analysis_collection.update_one(
            {"analysis_id": analysis_id, 
             "number": site_number},
            {"$set": {
                "cf_khz": cf_khz, 
                "threshold_db": thresh_db, 
                "cf_idx": cf,
                "threshold_idx": thresh,
                "bw10_khz": bw10_khz, 
                "bw20_khz": bw20_khz, 
                "bw30_khz": bw30_khz,
                "bw40_khz": bw40_khz,
                "bw10_idx": bw10,
                "bw20_idx": bw20, 
                "bw30_idx": bw30,
                "bw40_idx": bw40,
                "bw10_octave": bw10_octave,
                "bw20_octave": bw20_octave,
                "bw30_octave": bw30_octave, 
                "bw40_octave": bw40_octave,
                "continuous_bw_khz": continuous_bw_khz,
                "continuous_bw_idx": continuous_bw,
                "continuous_bw_octave": continuous_bw_octave,
                "onset_ms": onset, 
                "peak_ms": peak, 
                "offset_ms": offset,
                "peak_driven_rate_hz": peak_driven_rate,
                "marked": marked,
            }})

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
        self.gui_instance.plot_dict[self.map_number].bw10_idx = bw10
        self.gui_instance.plot_dict[self.map_number].bw20_idx = bw20
        self.gui_instance.plot_dict[self.map_number].bw30_idx = bw30
        self.gui_instance.plot_dict[self.map_number].bw40_idx = bw40
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
        smooth_tc, _, cf, thresh, bw10, bw20, bw30, bw40, continuous_bw, _ = \
            afunc.ttest_analyze_tuning_curve(
                afunc.ttest_driven_vs_spont_tc(*ttest_spike_counts))

        # Store analyzed data in the SitePlot object. 
        # Data is NOT saved until user hits 'Save' button
        self.densetc_plot.cf_idx = cf
        self.densetc_plot.thresh_idx = thresh
        self.densetc_plot.bw10_idx = bw10
        self.densetc_plot.bw20_idx = bw20
        self.densetc_plot.bw30_idx = bw30
        self.densetc_plot.bw40_idx = bw40
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
    def __init__(self):
        """
        Main application showing all Sites for a Map.
        Permits Auditory Field selection and overview of map properties and 
        analysis.
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

        self.ic_bool = False  # Changes coloring of histograms based on latency
        self.map_loaded = False
        self.subject_database = None
        self.map_metadata_collection = None
        self.map_metadata = None
        self.sites_collection = None
        self.densetc_analysis_collection = None
        self.bonus_analysis_collection = None
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
        self.analysis_id = ""

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
        self.map_file_path = ""
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
        self.open_file_button = Button(text="Load Map", size_hint=(1, 0.06))
        self.open_file_button.bind(on_release=self.load_map)
        self.open_ic_button = Button(text="Load IC", size_hint=(1, 0.06))
        self.open_ic_button.bind(on_release=self.load_ic)
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

        tools.add_widget(self.open_file_button)
        tools.add_widget(self.open_ic_button)
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
        self.toggle_contour.bind(on_release=self.on_toggle_contour)
        self.toggle_lineplot.bind(on_release=self.on_toggle_lineplot)
        self.toggle_bw.bind(on_release=self.on_toggle_bw)
        self.toggle_smooth.bind(on_release=self.on_toggle_smooth)
        self.toggle_heatmap.bind(on_release=self.on_toggle_heatmap)

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

        self.map_canvas = MapLayout(size_hint_x=None, size_hint_y=None)
        self.scroll = MapScroll(size_hint=(1, 1))
        self.scroll.add_widget(self.map_canvas)

        self.add_widget(tools)
        self.add_widget(self.scroll)
        self.add_widget(self.plot_tools_layout)

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

    def on_toggle_heatmap(self, _event):
        """Show spike heatmaps instead of default bubble plots."""
        if self.toggle_heatmap.state == "down":
            for plot in self.plot_dict.values():
                plot.use_heatmap = True
                plot.re_plot()
                plot.figure_canvas.draw()
        else:
            for plot in self.plot_dict.values():
                plot.use_heatmap = False
                plot.re_plot()
                plot.figure_canvas.draw()

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

    def on_psth_ylim(self, _spinner, text):
        """Changing PSTH ylim's. Useful to emphasize weakly responsive sites."""
        if text == "None":
            for plot in self.plot_dict.values():
                plot.re_plot()
                plot.figure_canvas.draw()
        else:
            for plot in self.plot_dict.values():
                plot.re_plot(min_y=int(text))
                plot.figure_canvas.draw()

    def on_toggle_contour(self, _event):
        """Display smoothed contour lines around TCs."""
        if self.toggle_contour.state == "down":
            for plot in self.plot_dict.values():
                plot.use_contour = True
                plot.re_plot()
                plot.figure_canvas.draw()
        else:
            for plot in self.plot_dict.values():
                plot.use_contour = False
                plot.re_plot()
                plot.figure_canvas.draw()

    def on_toggle_lineplot(self, _event):
        """Display spike counts as lines. Longer -> more spikes."""
        if self.toggle_lineplot.state == "down":
            for plot in self.plot_dict.values():
                plot.use_lineplot = True
                plot.re_plot()
                plot.figure_canvas.draw()
        else:
            for plot in self.plot_dict.values():
                plot.use_lineplot = False
                plot.re_plot()
                plot.figure_canvas.draw()

    def on_toggle_bw(self, _event):
        """Display 10-40 dB Bandwidths on top of TC."""
        if self.toggle_bw.state == "down":
            for plot in self.plot_dict.values():
                plot.use_bw = True
                plot.re_plot()
                plot.figure_canvas.draw()
        else:
            for plot in self.plot_dict.values():
                plot.use_bw = False
                plot.re_plot()
                plot.figure_canvas.draw()

    def on_toggle_smooth(self, _event):
        """Display smoothed TC instead of raw spike counts."""
        if self.toggle_smooth.state == "down":
            for plot in self.plot_dict.values():
                plot.use_smooth_tc = True
                plot.re_plot()
                plot.figure_canvas.draw()
        else:
            for plot in self.plot_dict.values():
                plot.use_smooth_tc = False
                plot.re_plot()
                plot.figure_canvas.draw()

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

            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("Success!", "Fields / Marks saved!")
            root.destroy()

    def increase_figsize(self, _event):
        """Increase matplotlib figure size."""
        for fig in self.plot_dict.values():
            fig.size = (fig.width / 0.75, fig.height / 0.75)

    def decrease_figsize(self, _event):
        """Decrease matplotlib figure size."""
        for fig in self.plot_dict.values():
            fig.size = (fig.width * 0.75, fig.height * 0.75)

    def load_map(self, _event):
        """Load cortical auditory map."""
        # TODO break out creating a new analysis as a separate function so I 
        #   don't repeat between cortical and IC funcs
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Load Data", 
                            "Select .json Map database for subject.")
        self.map_file_path = afunc.get_file(title="Select database JSON file", 
                                            filetypes=[("JSON", ".json")])
        if (self.map_file_path is None) or (self.map_file_path == ""):
            return
        # Initialize tinymongo database
        self.mongo_connection = TinyMongoClient(
            os.path.dirname(self.map_file_path))
        self.subject_database = getattr(self.mongo_connection,
                                        os.path.splitext(
                                            os.path.basename(
                                                self.map_file_path)
                                            )[0])
        self.map_metadata_collection = self.subject_database.metadata
        self.map_metadata = self.map_metadata_collection.find_one({})
        self.sites_collection = self.subject_database.sites
        self.densetc_data_collection = self.subject_database.densetc_data
        self.densetc_analysis_collection = \
            self.subject_database.densetc_analysis
        self.analysis_metadata_collection = \
            self.subject_database.analysis_metadata
        # TODO hacky way to duplicate be able to create new IC analysis with
        # new cortical analysis. Fix later.
        # IC version will replace with cortical one so that if creating new 
        # from IC then cortical is also created.
        # If there is no IC data/analysis (even a non-IC project) the 
        # collection is still created, but has no impact on the functionality 
        # of everything else. It will just be empty collections.
        self.bonus_analysis_collection = \
            self.subject_database.densetc_IC_analysis

        # Load project configuration information. Expect only 1 config result, 
        # or that all stored configs are redundant.
        # Currently just grabbing very first analysis that has a config (which 
        # for me is just the auto analysis program)
        # $exists is mongo operator I want, but it is not implemented in 
        # tinymongo. Using $ne is just a hacky way of doing the same thing -- 
        # find_one will only return documents given that the field 
        # 'configuration' exists. The operator $ne is always true against the 
        # data 'configuration' holds, so as long as the field exists in a
        # document, it will be returned. If a document doesn't have the field,
        # it will be skipped.
        self.project_configuration = \
            self.analysis_metadata_collection.find_one(
                {"configuration": {"$ne": False}})["configuration"]
        # Just in case it is unsorted
        self.frequency = np.sort(
            self.project_configuration["densetc_frequency_hz"])
        self.intensity = np.sort(
            self.project_configuration["densetc_intensity_db"])
        self.num_frequency = len(self.frequency)
        self.num_intensity = len(self.intensity)
        self.num_tones = self.project_configuration["densetc_num_tones"]

        # Grab voronoi data to draw map
        self.sites = [site for site in self.sites_collection.find({})]

        # Load an existing analysis to keep working on, or create a new 
        # analysis from an existing one
        # TODO Allow possibility of raw data analysis from scratch
        analysis_loaded = False
        while not analysis_loaded:
            analysis_selection, create_new_analysis = \
                afunc.load_analysis(self.analysis_metadata_collection)
            if analysis_selection is None:
                # Menu exited without selection
                return
            else:
                # load_analysis returns Series with analysis metadata, 
                # and whether or not to create a new analysis
                if create_new_analysis:
                    new_analysis_metadata = \
                        afunc.new_analysis_metadata_document()
                    if new_analysis_metadata is None:
                        # User hit cancel. Re-prompt to load analysis
                        continue
                    template_id = analysis_selection["_id"]
                    self.analysis_id = afunc.create_new_densetc_analysis(
                        template_id,
                        new_analysis_metadata,
                        self.analysis_metadata_collection,
                        self.densetc_analysis_collection,
                        self.bonus_analysis_collection)
                else:
                    self.analysis_id = analysis_selection["_id"]
                    analysis_loaded = True

        # Grab all data and analysis upfront and parse into dicts. MUCH faster 
        # than each site individually searching.
        # Keys are site numbers.
        self.densetc_data = {data["number"]: data for data in 
                             self.densetc_data_collection.find({})}
        self.densetc_analysis = {analysis["number"]: analysis for analysis in
                                 self.densetc_analysis_collection.find(
                                     {"analysis_id": self.analysis_id})}

        try:
            # TODO Allow loading new maps
            self.clear_map()
        except Exception as e:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("General error", 
                                 "Error occurred while trying to display map. "
                                 "Were the correct files selected?")
            logging.exception(e)
            root.destroy()

        self.display_map()
        print("\n *** Ready! *** \n")
        root.destroy()

    def load_ic(self, _event):
        """Load inferior-colliculus auditory 'map'."""
        # Identical to load_map except it handles IC data analysis
        # Instead of loading voronoi polygons, this creates a 'pseudo-map'
        # based on IC depth
        self.ic_bool = True
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Load Data", 
                            "Select .json IC database for subject.")
        self.map_file_path = afunc.get_file(title="Select database JSON file", 
                                            filetypes=[("JSON", ".json")])
        if (self.map_file_path is None) or (self.map_file_path == ""):
            return
        # Initialize tinymongo database
        self.mongo_connection = TinyMongoClient(
            os.path.dirname(self.map_file_path))
        self.subject_database = getattr(self.mongo_connection,
                                        os.path.splitext(
                                            os.path.basename(
                                                self.map_file_path))[0])
        self.map_metadata_collection = self.subject_database.metadata
        self.map_metadata = self.map_metadata_collection.find_one({})

        # For IC, map 'height' and 'width' are fabricated. Has no impact on
        # underlying map_metadata storage
        self.map_metadata["map_height"] = 3000
        self.map_metadata["map_width"] = 1000

        # 'sites' normally contain map number, xy and voronoi coords. 
        # For IC we fabricate polygons using 'depth' info
        self.densetc_data_collection = self.subject_database.densetc_IC_data
        ic_sites = [{"number": int(site["number"]), 
                     "depth": int(site["depth"])} for site in 
                    self.densetc_data_collection.find({})]
        ic_df = pd.DataFrame(ic_sites)
        ic_df = ic_df.sort_values("depth")
        ic_df = ic_df.reset_index(drop=True)
        pseudo_odd_x = 0.25
        pseudo_even_x = 0.75
        ic_df["x"] = ic_df["number"].apply(lambda x: 
            pseudo_odd_x if x % 2 else pseudo_even_x)
        ic_df["y"] = ic_df["depth"]
        ic_df["inter_depth"] = ic_df["y"].diff()
        # Typical IC map has 2x sites per depth, so inter_depth likely 
        # alternates 0, ~200, 0, ~200
        # To remove the zeros, but still allow for odd sites that don't have 2x 
        # sites per depth, we take max inter_depth per depth
        ic_df["inter_depth"] = ic_df["y"].apply(lambda x: 
            ic_df.loc[ic_df["y"] == x, "inter_depth"].max())
        # The first site(s) will have inter_depth==0 because nothing is before 
        # them to take a difference with
        # Changing the 0 to NaN allows us to backfill an inter_depth value from
        # the site(s) directly in front of them
        ic_df.loc[ic_df["inter_depth"] == 0, "inter_depth"] = np.nan
        ic_df["inter_depth"] = ic_df["inter_depth"].fillna(method="bfill")
        ic_df["vert_up"] = ic_df["y"] - (ic_df["inter_depth"] / 2)
        ic_df["vert_down"] = ic_df["y"] + (ic_df["inter_depth"] / 2)
        # Multiply 'depth coordinates' by -1 and then normalize between 
        # 0 and 1. When displayed, the most shallow sites (low-freq IC) will 
        # correctly be at the top, and the deepest (high-freq) at the bottom
        ic_df[["y", "vert_up", "vert_down"]] = -ic_df[
            ["y", "vert_up", "vert_down"]]
        min_coord = ic_df["vert_down"].min()
        max_coord = ic_df["vert_up"].max()
        ic_df[["y", "vert_up", "vert_down"]] = (
            (ic_df[["y", "vert_up", "vert_down"]] - min_coord) / 
            (max_coord - min_coord))

        ic_sites = ic_df.to_dict("records")

        self.sites = []
        for site in ic_sites:
            site["voronoi_centroid"] = [site["x"], site["y"]]
            if site["x"] == pseudo_odd_x:
                site["voronoi_vertices"] = [(0, site["vert_down"]), 
                                            (0, site["vert_up"]),
                                            (0.5, site["vert_up"]), 
                                            (0.5, site["vert_down"])]
            else:
                site["voronoi_vertices"] = [(0.5, site["vert_down"]),
                                            (0.5, site["vert_up"]),
                                            (1, site["vert_up"]), 
                                            (1, site["vert_down"])]

            self.sites.append(site)

        self.densetc_analysis_collection = \
            self.subject_database.densetc_IC_analysis
        self.analysis_metadata_collection = \
            self.subject_database.analysis_metadata
        # TODO Fix. See cortical loading todo above
        # Cortical one loads IC as bonus
        self.bonus_analysis_collection = self.subject_database.densetc_analysis

        self.project_configuration = \
            self.analysis_metadata_collection.find_one(
                {"configuration": {"$ne": False}})["configuration"]
        # Just in case it is unsorted
        self.frequency = np.sort(
            self.project_configuration["densetc_frequency_hz"])
        self.intensity = np.sort(
            self.project_configuration["densetc_intensity_db"])
        self.num_frequency = len(self.frequency)
        self.num_intensity = len(self.intensity)
        self.num_tones = self.project_configuration["densetc_num_tones"]

        # Load an existing analysis to keep working on, or create a new 
        # analysis from an existing one
        # TODO Allow possibility of raw data analysis from scratch
        analysis_loaded = False
        while not analysis_loaded:
            analysis_selection, create_new_analysis = \
                afunc.load_analysis(self.analysis_metadata_collection)
            if analysis_selection is None:
                # Menu exited without selection
                return
            else:
                # load_analysis returns Series with analysis metadata, 
                # and whether or not to create a new analysis
                if create_new_analysis:
                    new_analysis_metadata = \
                        afunc.new_analysis_metadata_document()
                    if new_analysis_metadata is None:
                        # User hit cancel. Re-prompt to load analysis
                        continue
                    template_id = analysis_selection["_id"]
                    self.analysis_id = afunc.create_new_densetc_analysis(
                        template_id,
                        new_analysis_metadata,
                        self.analysis_metadata_collection,
                        self.densetc_analysis_collection,
                        self.bonus_analysis_collection)
                else:
                    self.analysis_id = analysis_selection["_id"]
                    analysis_loaded = True

        # Grab all data and analysis upfront and parse into dicts. 
        # MUCH faster than each site individually searching.
        # Keys are site numbers.
        self.densetc_data = {data["number"]: data for data in 
                             self.densetc_data_collection.find({})}
        self.densetc_analysis = {analysis["number"]: analysis for analysis in
                                 self.densetc_analysis_collection.find(
                                     {"analysis_id": self.analysis_id})}

        try:
            # TODO Allow loading new maps
            self.clear_map()
        except Exception as e:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("General error", 
                                 "Error occurred while trying to display map. "
                                 "Were the correct files selected?")
            logging.exception(e)
            root.destroy()

        self.display_map()
        print("\n *** Ready! *** \n")
        root.destroy()

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

    def clear_map(self):
        pass  # TODO allow loading new maps without having to restart GUI / CLI


class MapLayout(FloatLayout):
    def on_touch_down(self, touch):
        """
        First checks if double-tap: If yes, check if over a map site, and pull 
        up the respective details screen.
        Checks by building a polygon for each site using current line points 
        and shapely to check if mouse touch fell in any of them. Must be done 
        on the fly to accommodate zoom in/out capabilities.
        If not a double-click or outside of site polygons:
        Checks if Select/Deselect A1 is toggled: If yes, and user is not 
        scrolling (based on mouse-move delay), then draw a red line following 
        user's mouse. 
        If no, let GUI process input normally (Kivy handles event).
        """
        if touch.is_double_tap:
            for line_ in self.parent.parent.vor_lines.values():
                poly_points = line_.line.points
                poly_x = poly_points[0:][::2]
                poly_y = poly_points[1:][::2]
                shapely_poly = Polygon(zip(poly_x, poly_y))
                if shapely_poly.contains(Point(touch.x, touch.y)):
                    # If event occurred over a site, pull up that site's 
                    # detailed plot
                    num = line_.site_number
                    screen_manager = self.parent.parent.parent.manager
                    screen_manager.switch_to(
                        self.parent.parent.site_screens[num])

        if ((self.parent.parent.toggle.state == "down") or 
            (self.parent.parent.deselect_toggle.state == "down") or
            (self.parent.parent.show_figure_toggle.state == "down") or 
            (self.parent.parent.hide_figure_toggle.state == "down")):
            with self.canvas:
                Color(1, 0, 0)
                touch.ud["line"] = Line(points=(touch.x, touch.y), width=1.5)
        else:
            super(MapLayout, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        """
        Checks if Select/Deselect A1 is toggled: If yes, and on_touch_down 
        generated a paint line, then continue drawing line at user's mouse 
        location.
        Using try/except because occasionally an error is thrown by GUI (do not
        know why) and will crash the program.
        Ignoring the error has no consequence, so the except clause simply 
        passes.
        """
        try:
            if ((self.parent.parent.toggle.state == "down") or 
                (self.parent.parent.deselect_toggle.state == "down") or
                (self.parent.parent.show_figure_toggle.state == "down") or 
                (self.parent.parent.hide_figure_toggle.state == "down")):
                touch.ud["line"].points += [touch.x, touch.y]
        except:  # TODO Figure out error raised
            pass

    def on_touch_up(self, touch):
        """
        on_touch_up handles three user interaction cases:
        1) For some reason, Kivy handles mouse scrolling as on_touch_up events 
            when using a ScrollView.
            Mouse scrolling scales the view up or down, respectively
        2) If Select/Deselect A1 is toggled and user was drawing a line: Gather
            the points, check for any that fall within Voronoi cells, and 
            update A1 selection accordingly. Mesh colors are also updated to 
            reflect selections. Finally, delete the line the user drew.
        3) None of the above. Kivy handles the touch internally
        """
        if touch.is_mouse_scrolling:
            h = self.height
            w = self.width
            if touch.button == "scrollup":
                self.size = (w * 0.9, h * 0.9)
            elif touch.button == "scrolldown":
                self.size = (w * 1.1, h * 1.1)
            return True

        elif ((self.parent.parent.toggle.state == "down") or 
              (self.parent.parent.deselect_toggle.state == "down") or
              (self.parent.parent.show_figure_toggle.state == "down") or 
              (self.parent.parent.hide_figure_toggle.state == "down")):
            try:
                selection_points = [Point(x, y) for x, y in zip(
                    touch.ud["line"].points[0:][::2],
                    touch.ud["line"].points[1:][::2])]
            except KeyError:
                # Error sometimes thrown when program tries to interpret a line
                # drawn over other GUI elements
                super(MapLayout, self).on_touch_up(touch)
                return

            for line_ in self.parent.parent.vor_lines.values():
                poly_points = line_.line.points
                poly_x = poly_points[0:][::2]
                poly_y = poly_points[1:][::2]
                shapely_poly = Polygon(zip(poly_x, poly_y))
                if list(filter(shapely_poly.contains, selection_points)):
                    site_num = line_.site_number
                    # Check if cell is currently interactive or hidden. 
                    # If hidden, move on
                    if not self.parent.parent.vor_active[site_num]:
                        continue
                    # Update mesh colors
                    if self.parent.parent.toggle.state == "down":
                        line_.line.width = 3
                        field_selection = self.parent.parent.field_spinner.text
                        for field in self.parent.parent.fields:
                            if field == field_selection:
                                self.parent.parent.map_sets[field].add(site_num)
                                self.parent.parent.vor_meshes[site_num].color.rgb = \
                                    hex2rgb(self.parent.parent.field_colors[field])
                                alpha_value = self.parent.parent.field_alpha_slider.value_normalized
                                self.parent.parent.vor_meshes[site_num].color.a = alpha_value
                                line_.color.rgb = hex2rgb(self.parent.parent.field_line_colors[field])
                                # Allow user to progressively select and hide sites for a field
                                # Pass fake event 'field_selection' to satisfy Kivy callback requirements
                                self.parent.parent.on_hide_field(
                                    "field_selection", site_number=site_num)
                            else:
                                if self.parent.parent.marks_active:
                                    # If selecting Marks, don't remove Sites from other Field sets
                                    continue
                                elif field == "Mark":
                                    # If selecting Fields, don't remove Marks from map set
                                    continue

                                # Prevent duplicate field assignments
                                try:
                                    self.parent.parent.map_sets[field].remove(
                                        site_num)
                                except KeyError:  # If site is not in set, skip
                                    pass

                    # 'Deselect' is toggled on
                    elif self.parent.parent.deselect_toggle.state == "down":
                        # xkcd:steel grey
                        line_.color.rgb = [0.435, 0.51, 0.541]
                        line_.line.width = 1.5
                        self.parent.parent.vor_meshes[site_num].color.rgb = [1, 1, 1]
                        alpha_value = self.parent.parent.field_alpha_slider.value_normalized
                        self.parent.parent.vor_meshes[site_num].color.a = alpha_value
                        for field in self.parent.parent.fields:
                            if (field == "Mark" and 
                                not self.parent.parent.marks_active):
                                # Skip removing Site from Mark map set if not working on Marks
                                continue
                            elif (field != "Mark" and 
                                  self.parent.parent.marks_active):
                                # Skip removing Site from Fields sets if not working on Fields
                                continue
                            try:
                                self.parent.parent.map_sets[field].remove(site_num)
                            except KeyError:  # If site is not in set, skip
                                pass
                    elif self.parent.parent.show_figure_toggle.state == "down":
                        self.parent.parent.plot_dict[site_num].active = True
                        self.parent.parent.plot_dict[site_num].opacity = 1
                        self.parent.parent.plot_dict[site_num].manually_hidden = False
                    elif self.parent.parent.hide_figure_toggle.state == "down":
                        self.parent.parent.plot_dict[site_num].active = False
                        self.parent.parent.plot_dict[site_num].opacity = 0
                        self.parent.parent.plot_dict[site_num].manually_hidden = True

            self.canvas.remove(touch.ud["line"])
        else:
            super(MapLayout, self).on_touch_up(touch)


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
        self.cf_cmap = matplotlib.cm.get_cmap(kwargs["cf_cmap"])
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
        self.bw10_idx = self.site_analysis["bw10_idx"].copy()
        self.saved_bw10_idx = self.site_analysis["bw10_idx"].copy()
        self.bw20_idx = self.site_analysis["bw20_idx"].copy()
        self.saved_bw20_idx = self.site_analysis["bw20_idx"].copy()
        self.bw30_idx = self.site_analysis["bw30_idx"].copy()
        self.saved_bw30_idx = self.site_analysis["bw30_idx"].copy()
        self.bw40_idx = self.site_analysis["bw40_idx"].copy()
        self.saved_bw40_idx = self.site_analysis["bw40_idx"].copy()
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
        self.bw10_line = None
        self.bw20_line = None
        self.bw30_line = None
        self.bw40_line = None
        self.bw10_markers = [None, None]
        self.bw20_markers = [None, None]
        self.bw30_markers = [None, None]
        self.bw40_markers = [None, None]
        self.bw10_press = [0, 0]
        self.bw20_press = [0, 0]
        self.bw30_press = [0, 0]
        self.bw40_press = [0, 0]
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
        self.cf_cmap = matplotlib.cm.get_cmap(cf_cmap)
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
            if np.any(xdata < self.offset_line.get_xdata()):
                self.onset_line.set_xdata(xdata)
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
            if np.any(xdata > self.onset_line.get_xdata()):
                self.offset_line.set_xdata(xdata)
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

        self.bubble.update({"offsets": list(zip(x, y)), "sizes": scaled_s ** 2})

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
                                 lw=0.5, c=color)
        ax.set_facecolor(axis_color)

        if self.use_bw and (self.cf_idx is not None):
            # Assumes 5dB step size. Fix if it ever changes
            bw10_y = self.thresh_idx + 2
            bw20_y = self.thresh_idx + 4
            bw30_y = self.thresh_idx + 6
            bw40_y = self.thresh_idx + 8
            if self.bw10_idx[0] is not None:
                self.bw10_line = ax.plot(self.bw10_idx, [bw10_y, bw10_y], 
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw10_markers[0] = ax.plot(self.bw10_idx[0], bw10_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw10_markers[1] = ax.plot(self.bw10_idx[1], bw10_y, 
                                                   "rd", ms=8, picker=5)[0]
            if self.bw20_idx[0] is not None:
                self.bw20_line = ax.plot(self.bw20_idx, [bw20_y, bw20_y], 
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw20_markers[0] = ax.plot(self.bw20_idx[0], bw20_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw20_markers[1] = ax.plot(self.bw20_idx[1], bw20_y, 
                                                   "rd", ms=8, picker=5)[0]
            if self.bw30_idx[0] is not None:
                self.bw30_line = ax.plot(self.bw30_idx, [bw30_y, bw30_y], 
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw30_markers[0] = ax.plot(self.bw30_idx[0], bw30_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw30_markers[1] = ax.plot(self.bw30_idx[1], bw30_y, 
                                                   "rd", ms=8, picker=5)[0]
            if self.bw40_idx[0] is not None:
                self.bw40_line = ax.plot(self.bw40_idx, [bw40_y, bw40_y], 
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw40_markers[0] = ax.plot(self.bw40_idx[0], bw40_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw40_markers[1] = ax.plot(self.bw40_idx[1], bw40_y, 
                                                   "rd", ms=8, picker=5)[0]

        if self.use_contour:
            self.contour_line = ax.contour(self.contour_tc, levels=[0])

        if self.cf_idx is not None:
            self.cf_marker = ax.plot(self.cf_idx, self.thresh_idx, 
                                     "r*", ms=8, alpha=0.5)[0]
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

        if self.use_bw and self.cf_idx:
            # Assumes 5dB step size. Fix if it ever changes
            bw10_y = self.thresh_idx + 2
            bw20_y = self.thresh_idx + 4
            bw30_y = self.thresh_idx + 6
            bw40_y = self.thresh_idx + 8
            if self.bw10_idx[0] is not None:
                self.bw10_line = ax.plot(self.bw10_idx, [bw10_y, bw10_y], 
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw10_markers[0] = ax.plot(self.bw10_idx[0], bw10_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw10_markers[1] = ax.plot(self.bw10_idx[1], bw10_y, 
                                                   "rd", ms=8, picker=5)[0]
            if self.bw20_idx[0] is not None:
                self.bw20_line = ax.plot(self.bw20_idx, [bw20_y, bw20_y], 
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw20_markers[0] = ax.plot(self.bw20_idx[0], bw20_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw20_markers[1] = ax.plot(self.bw20_idx[1], bw20_y, 
                                                   "rd", ms=8, picker=5)[0]
            if self.bw30_idx[0] is not None:
                self.bw30_line = ax.plot(self.bw30_idx, [bw30_y, bw30_y], 
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw30_markers[0] = ax.plot(self.bw30_idx[0], bw30_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw30_markers[1] = ax.plot(self.bw30_idx[1], bw30_y,
                                                   "rd", ms=8, picker=5)[0]
            if self.bw40_idx[0] is not None:
                self.bw40_line = ax.plot(self.bw40_idx, [bw40_y, bw40_y],
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw40_markers[0] = ax.plot(self.bw40_idx[0], bw40_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw40_markers[1] = ax.plot(self.bw40_idx[1], bw40_y, 
                                                   "rd", ms=8, picker=5)[0]

        if self.use_contour:
            if self.detailed_plot:
                self.contour_line = ax.contour(self.contour_tc, levels=[0], 
                                               colors="xkcd:white")
            else:
                self.contour_line = ax.contour(self.contour_tc, levels=[0])

        if self.cf_idx:
            self.cf_marker = ax.plot(self.cf_idx, self.thresh_idx, 
                                     "r*", ms=8, alpha=1)[0]
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

        if self.use_bw and self.cf_idx:
            # TODO Assumes 5dB step size. Fix if it ever changes
            bw10_y = self.thresh_idx + 2
            bw20_y = self.thresh_idx + 4
            bw30_y = self.thresh_idx + 6
            bw40_y = self.thresh_idx + 8
            if self.bw10_idx[0] is not None:
                self.bw10_line = ax.plot(self.bw10_idx, [bw10_y, bw10_y], 
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw10_markers[0] = ax.plot(self.bw10_idx[0], bw10_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw10_markers[1] = ax.plot(self.bw10_idx[1], bw10_y, 
                                                   "rd", ms=8, picker=5)[0]
            if self.bw20_idx[0] is not None:
                self.bw20_line = ax.plot(self.bw20_idx, [bw20_y, bw20_y], 
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw20_markers[0] = ax.plot(self.bw20_idx[0], bw20_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw20_markers[1] = ax.plot(self.bw20_idx[1], bw20_y, 
                                                   "rd", ms=8, picker=5)[0]
            if self.bw30_idx[0] is not None:
                self.bw30_line = ax.plot(self.bw30_idx, [bw30_y, bw30_y], 
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw30_markers[0] = ax.plot(self.bw30_idx[0], bw30_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw30_markers[1] = ax.plot(self.bw30_idx[1], bw30_y, 
                                                   "rd", ms=8, picker=5)[0]
            if self.bw40_idx[0] is not None:
                self.bw40_line = ax.plot(self.bw40_idx, [bw40_y, bw40_y], 
                                         "r", lw=1.5)[0]
                if self.detailed_plot:
                    self.bw40_markers[0] = ax.plot(self.bw40_idx[0], bw40_y, 
                                                   "rd", ms=8, picker=5)[0]
                    self.bw40_markers[1] = ax.plot(self.bw40_idx[1], bw40_y, 
                                                   "rd", ms=8, picker=5)[0]

        if self.use_contour:
            if self.detailed_plot:
                self.contour_line = ax.contour(self.contour_tc, levels=[0], 
                                               colors="xkcd:white")
            else:
                self.contour_line = ax.contour(self.contour_tc, levels=[0])

        if self.cf_idx:
            self.cf_marker = ax.plot(self.cf_idx, self.thresh_idx, 
                                     "r*", ms=8, alpha=1)[0]
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
            self.cf_marker.set_xdata(self.cf_idx)
            self.cf_marker.set_ydata(self.thresh_idx)
        # Ensure all BWs are set correctly. If BW exceeds range, set to None. 
        # If setting new BW (was previously None, but now should exist with new
        # threshold), set to very wide default range across available frequency
        # range
        # Assumes 5dB step size. Fix if it ever changes
        freq_range = self.gui_instance.num_frequency
        int_range = self.gui_instance.num_intensity
        if (self.thresh_idx + 2) <= int_range:
            if self.bw10_idx[0] is None:
                self.bw10_idx = [10, freq_range - 10]
        else:
            self.bw10_idx = [None, None]
        if (self.thresh_idx + 4) <= int_range:
            if self.bw20_idx[0] is None:
                self.bw20_idx = [10, freq_range - 10]
        else:
            self.bw20_idx = [None, None]
        if (self.thresh_idx + 6) <= int_range:
            if self.bw30_idx[0] is None:
                self.bw30_idx = [10, freq_range - 10]
        else:
            self.bw30_idx = [None, None]
        if (self.thresh_idx + 8) <= int_range:
            if self.bw40_idx[0] is None:
                self.bw40_idx = [10, freq_range - 10]
        else:
            self.bw40_idx = [None, None]

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
        if self.bw10_markers[0] is not None:
            if self.bw10_markers[0].contains(event)[0]:
                self.bw_pressed = True
                self.bw10_markers[0].set_ms(12)
                self.bw10_press[0] = 1
                event.canvas.draw()
            elif self.bw10_markers[1].contains(event)[0]:
                self.bw_pressed = True
                self.bw10_markers[1].set_ms(12)
                self.bw10_press[1] = 1
                event.canvas.draw()
        if self.bw20_markers[0] is not None:
            if self.bw20_markers[0].contains(event)[0]:
                self.bw_pressed = True
                self.bw20_markers[0].set_ms(12)
                self.bw20_press[0] = 1
                event.canvas.draw()
            elif self.bw20_markers[1].contains(event)[0]:
                self.bw_pressed = True
                self.bw20_markers[1].set_ms(12)
                self.bw20_press[1] = 1
                event.canvas.draw()
        if self.bw30_markers[0] is not None:
            if self.bw30_markers[0].contains(event)[0]:
                self.bw_pressed = True
                self.bw30_markers[0].set_ms(12)
                self.bw30_press[0] = 1
                event.canvas.draw()
            elif self.bw30_markers[1].contains(event)[0]:
                self.bw_pressed = True
                self.bw30_markers[1].set_ms(12)
                self.bw30_press[1] = 1
                event.canvas.draw()
        if self.bw40_markers[0] is not None:
            if self.bw40_markers[0].contains(event)[0]:
                self.bw_pressed = True
                self.bw40_markers[0].set_ms(12)
                self.bw40_press[0] = 1
                event.canvas.draw()
            elif self.bw40_markers[1].contains(event)[0]:
                self.bw_pressed = True
                self.bw40_markers[1].set_ms(12)
                self.bw40_press[1] = 1
                event.canvas.draw()

    def move_bw(self, event_x, event_y):
        """Ongoing UX for user updating bandwidth."""
        # Must transform into axes user data coordinates (xlim, ylim) in order 
        # to move line to appropriate x-coordinate based on mouse position
        ax_inv = self.ax[1].transData.inverted()
        xdata, ydata = ax_inv.transform((event_x, event_y))
        if xdata is None:
            return

        if self.bw10_press[0]:
            if xdata < self.bw10_markers[1].get_xdata():
                # Don't allow markers to cross each other
                if xdata >= 0:
                    # Limit BW to lowest frequency (index 0)
                    x = int(round(xdata))
                else:
                    x = 0
                self.bw10_markers[0].set_xdata(x)
                self.bw10_idx[0] = x
                self.bw10_line.set_xdata(self.bw10_idx)
                if self.detailed_plot:
                    self.on_changes_signal.send()
        if self.bw10_press[1]:
            if xdata > self.bw10_markers[0].get_xdata():
                # Don't allow markers to cross each other
                if xdata <= (self.gui_instance.num_frequency - 1):
                    # Limit BW to highest frequency (max index)
                    x = int(round(xdata))
                else:
                    x = self.gui_instance.num_frequency - 1
                self.bw10_markers[1].set_xdata(x)
                self.bw10_idx[1] = x
                self.bw10_line.set_xdata(self.bw10_idx)
                if self.detailed_plot:
                    self.on_changes_signal.send()

        if self.bw20_press[0]:
            if xdata < self.bw20_markers[1].get_xdata():
                # Don't allow markers to cross each other
                if xdata >= 0:
                    # Limit BW to lowest frequency (index 0)
                    x = int(round(xdata))
                else:
                    x = 0
                self.bw20_markers[0].set_xdata(x)
                self.bw20_idx[0] = x
                self.bw20_line.set_xdata(self.bw20_idx)
                if self.detailed_plot:
                    self.on_changes_signal.send()
        if self.bw20_press[1]:
            if xdata > self.bw20_markers[0].get_xdata():
                # Don't allow markers to cross each other
                if xdata <= (self.gui_instance.num_frequency - 1):
                    # Limit BW to highest frequency (max index)
                    x = int(round(xdata))
                else:
                    x = self.gui_instance.num_frequency - 1
                self.bw20_markers[1].set_xdata(x)
                self.bw20_idx[1] = x
                self.bw20_line.set_xdata(self.bw20_idx)
                if self.detailed_plot:
                    self.on_changes_signal.send()

        if self.bw30_press[0]:
            if xdata < self.bw30_markers[1].get_xdata():
                # Don't allow markers to cross each other
                if xdata >= 0:
                    # Limit BW to lowest frequency (index 0)
                    x = int(round(xdata))
                else:
                    x = 0
                self.bw30_markers[0].set_xdata(x)
                self.bw30_idx[0] = x
                self.bw30_line.set_xdata(self.bw30_idx)
                if self.detailed_plot:
                    self.on_changes_signal.send()
        if self.bw30_press[1]:
            if xdata > self.bw30_markers[0].get_xdata():
                # Don't allow markers to cross each other
                if xdata <= (self.gui_instance.num_frequency - 1):
                    # Limit BW to highest frequency (max index)
                    x = int(round(xdata))
                else:
                    x = self.gui_instance.num_frequency - 1
                self.bw30_markers[1].set_xdata(x)
                self.bw30_idx[1] = x
                self.bw30_line.set_xdata(self.bw30_idx)
                if self.detailed_plot:
                    self.on_changes_signal.send()

        if self.bw40_press[0]:
            if xdata < self.bw40_markers[1].get_xdata():
                # Don't allow markers to cross each other
                if xdata >= 0:
                    # Limit BW to lowest frequency (index 0)
                    x = int(round(xdata))
                else:
                    x = 0
                self.bw40_markers[0].set_xdata(x)
                self.bw40_idx[0] = x
                self.bw40_line.set_xdata(self.bw40_idx)
                if self.detailed_plot:
                    self.on_changes_signal.send()
        if self.bw40_press[1]:
            if xdata > self.bw40_markers[0].get_xdata():
                # Don't allow markers to cross each other
                if xdata <= (self.gui_instance.num_frequency - 1):
                    # Limit BW to highest frequency (max index)
                    x = int(round(xdata))
                else:
                    x = self.gui_instance.num_frequency - 1
                self.bw40_markers[1].set_xdata(x)
                self.bw40_idx[1] = x
                self.bw40_line.set_xdata(self.bw40_idx)
                if self.detailed_plot:
                    self.on_changes_signal.send()

        self.figure_canvas.draw()

    def off_bw(self):
        """Final UX for user updating bandwidth."""
        self.bw10_press = [0, 0]
        self.bw20_press = [0, 0]
        self.bw30_press = [0, 0]
        self.bw40_press = [0, 0]
        if self.bw10_markers[0] is not None:
            self.bw10_markers[0].set_ms(8)
            self.bw10_markers[1].set_ms(8)
        if self.bw20_markers[0] is not None:
            self.bw20_markers[0].set_ms(8)
            self.bw20_markers[1].set_ms(8)
        if self.bw30_markers[0] is not None:
            self.bw30_markers[0].set_ms(8)
            self.bw30_markers[1].set_ms(8)
        if self.bw40_markers[0] is not None:
            self.bw40_markers[0].set_ms(8)
            self.bw40_markers[1].set_ms(8)

        self.figure_canvas.draw()


if __name__ == "__main__":
    FieldSelectionApp().run()
    App.stop()
