# Matplotlib

## matplotlib.pyplot

pyplot是一个**基于状态**的matplotlib接口。它提供了一种类似MATLAB的绘图方法。pyplot中维护当前状态（figure，axes等）

在matplotlib库里，总分成两种绘图方法

1. 方法一：**函数式绘图**
2. 方法二：**面向对象式绘图**

```python
# 函数式
import matplotlib.pyplot as plt
plt.plot([1,2,3],[4,6,5])
plt.title("My title")
plt.show()

# 面向对象
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1,2,3],[4,6,5])
ax.set_title("My Title")
plt.show()
```





### API

| Function                                                     | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`acorr`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.acorr.html#matplotlib.pyplot.acorr) | Plot the autocorrelation of *x*.                             |
| [`angle_spectrum`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.angle_spectrum.html#matplotlib.pyplot.angle_spectrum) | Plot the angle spectrum.                                     |
| [`annotate`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.annotate.html#matplotlib.pyplot.annotate) | Annotate the point *xy* with text *text*.                    |
| [`arrow`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.arrow.html#matplotlib.pyplot.arrow) | Add an arrow to the axes.                                    |
| [`autoscale`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.autoscale.html#matplotlib.pyplot.autoscale) | Autoscale the axis view to the data (toggle).                |
| [`axes`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axes.html#matplotlib.pyplot.axes) | Add an axes to the current figure and make it the current axes. |
| [`axhline`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axhline.html#matplotlib.pyplot.axhline) | Add a horizontal line across the axis.                       |
| [`axhspan`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axhspan.html#matplotlib.pyplot.axhspan) | Add a horizontal span (rectangle) across the axis.           |
| [`axis`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axis.html#matplotlib.pyplot.axis) | Convenience method to get or set some axis properties.       |
| [`axvline`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axvline.html#matplotlib.pyplot.axvline) | Add a vertical line across the axes.                         |
| [`axvspan`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axvspan.html#matplotlib.pyplot.axvspan) | Add a vertical span (rectangle) across the axes.             |
| [`bar`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.bar.html#matplotlib.pyplot.bar) | Make a bar plot.                                             |
| [`barbs`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.barbs.html#matplotlib.pyplot.barbs) | Plot a 2D field of barbs.                                    |
| [`barh`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.barh.html#matplotlib.pyplot.barh) | Make a horizontal bar plot.                                  |
| [`box`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.box.html#matplotlib.pyplot.box) | Turn the axes box on or off on the current axes.             |
| [`boxplot`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.boxplot.html#matplotlib.pyplot.boxplot) | Make a box and whisker plot.                                 |
| [`broken_barh`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.broken_barh.html#matplotlib.pyplot.broken_barh) | Plot a horizontal sequence of rectangles.                    |
| [`cla`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.cla.html#matplotlib.pyplot.cla) | Clear the current axes.                                      |
| [`clabel`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.clabel.html#matplotlib.pyplot.clabel) | Label a contour plot.                                        |
| [`clf`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.clf.html#matplotlib.pyplot.clf) | Clear the current figure.                                    |
| [`clim`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.clim.html#matplotlib.pyplot.clim) | Set the color limits of the current image.                   |
| [`close`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.close.html#matplotlib.pyplot.close) | Close a figure window.                                       |
| [`cohere`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.cohere.html#matplotlib.pyplot.cohere) | Plot the coherence between *x* and *y*.                      |
| [`colorbar`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.colorbar.html#matplotlib.pyplot.colorbar) | Add a colorbar to a plot.                                    |
| [`contour`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contour.html#matplotlib.pyplot.contour) | Plot contours.                                               |
| [`contourf`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html#matplotlib.pyplot.contourf) | Plot contours.                                               |
| [`csd`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.csd.html#matplotlib.pyplot.csd) | Plot the cross-spectral density.                             |
| [`delaxes`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.delaxes.html#matplotlib.pyplot.delaxes) | Remove the `Axes` *ax* (defaulting to the current axes) from its figure. |
| [`draw`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.draw.html#matplotlib.pyplot.draw) | Redraw the current figure.                                   |
| [`errorbar`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.errorbar.html#matplotlib.pyplot.errorbar) | Plot y versus x as lines and/or markers with attached errorbars. |
| [`eventplot`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.eventplot.html#matplotlib.pyplot.eventplot) | Plot identical parallel lines at the given positions.        |
| [`figimage`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figimage.html#matplotlib.pyplot.figimage) | Add a non-resampled image to the figure.                     |
| [`figlegend`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figlegend.html#matplotlib.pyplot.figlegend) | Place a legend on the figure.                                |
| [`fignum_exists`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.fignum_exists.html#matplotlib.pyplot.fignum_exists) | Return whether the figure with the given id exists.          |
| [`figtext`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figtext.html#matplotlib.pyplot.figtext) | Add text to figure.                                          |
| [`figure`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure) | Create a new figure.                                         |
| [`fill`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.fill.html#matplotlib.pyplot.fill) | Plot filled polygons.                                        |
| [`fill_between`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.fill_between.html#matplotlib.pyplot.fill_between) | Fill the area between two horizontal curves.                 |
| [`fill_betweenx`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.fill_betweenx.html#matplotlib.pyplot.fill_betweenx) | Fill the area between two vertical curves.                   |
| [`findobj`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.findobj.html#matplotlib.pyplot.findobj) | Find artist objects.                                         |
| [`gca`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.gca.html#matplotlib.pyplot.gca) | Get the current [`Axes`](https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes) instance on the current figure matching the given keyword args, or create one. |
| [`gcf`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.gcf.html#matplotlib.pyplot.gcf) | Get the current figure.                                      |
| [`gci`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.gci.html#matplotlib.pyplot.gci) | Get the current colorable artist.                            |
| [`get_figlabels`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.get_figlabels.html#matplotlib.pyplot.get_figlabels) | Return a list of existing figure labels.                     |
| [`get_fignums`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.get_fignums.html#matplotlib.pyplot.get_fignums) | Return a list of existing figure numbers.                    |
| [`grid`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.grid.html#matplotlib.pyplot.grid) | Configure the grid lines.                                    |
| [`hexbin`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hexbin.html#matplotlib.pyplot.hexbin) | Make a 2D hexagonal binning plot of points *x*, *y*.         |
| [`hist`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html#matplotlib.pyplot.hist) | Plot a histogram.                                            |
| [`hist2d`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist2d.html#matplotlib.pyplot.hist2d) | Make a 2D histogram plot.                                    |
| [`hlines`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hlines.html#matplotlib.pyplot.hlines) | Plot horizontal lines at each *y* from *xmin* to *xmax*.     |
| [`imread`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imread.html#matplotlib.pyplot.imread) | Read an image from a file into an array.                     |
| [`imsave`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imsave.html#matplotlib.pyplot.imsave) | Save an array as an image file.                              |
| [`imshow`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html#matplotlib.pyplot.imshow) | Display an image, i.e.                                       |
| [`install_repl_displayhook`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.install_repl_displayhook.html#matplotlib.pyplot.install_repl_displayhook) | Install a repl display hook so that any stale figure are automatically redrawn when control is returned to the repl. |
| [`ioff`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ioff.html#matplotlib.pyplot.ioff) | Turn the interactive mode off.                               |
| [`ion`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ion.html#matplotlib.pyplot.ion) | Turn the interactive mode on.                                |
| [`isinteractive`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.isinteractive.html#matplotlib.pyplot.isinteractive) | Return the status of interactive mode.                       |
| [`legend`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend) | Place a legend on the axes.                                  |
| [`locator_params`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.locator_params.html#matplotlib.pyplot.locator_params) | Control behavior of major tick locators.                     |
| [`loglog`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.loglog.html#matplotlib.pyplot.loglog) | Make a plot with log scaling on both the x and y axis.       |
| [`magnitude_spectrum`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.magnitude_spectrum.html#matplotlib.pyplot.magnitude_spectrum) | Plot the magnitude spectrum.                                 |
| [`margins`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.margins.html#matplotlib.pyplot.margins) | Set or retrieve autoscaling margins.                         |
| [`matshow`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.matshow.html#matplotlib.pyplot.matshow) | Display an array as a matrix in a new figure window.         |
| [`minorticks_off`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.minorticks_off.html#matplotlib.pyplot.minorticks_off) | Remove minor ticks from the axes.                            |
| [`minorticks_on`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.minorticks_on.html#matplotlib.pyplot.minorticks_on) | Display minor ticks on the axes.                             |
| [`pause`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pause.html#matplotlib.pyplot.pause) | Pause for *interval* seconds.                                |
| [`pcolor`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolor.html#matplotlib.pyplot.pcolor) | Create a pseudocolor plot with a non-regular rectangular grid. |
| [`pcolormesh`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolormesh.html#matplotlib.pyplot.pcolormesh) | Create a pseudocolor plot with a non-regular rectangular grid. |
| [`phase_spectrum`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.phase_spectrum.html#matplotlib.pyplot.phase_spectrum) | Plot the phase spectrum.                                     |
| [`pie`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pie.html#matplotlib.pyplot.pie) | Plot a pie chart.                                            |
| [`plot`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot) | Plot y versus x as lines and/or markers.                     |
| [`plot_date`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot_date.html#matplotlib.pyplot.plot_date) | Plot data that contains dates.                               |
| [`plotfile`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plotfile.html#matplotlib.pyplot.plotfile) | Plot the data in a file.                                     |
| [`polar`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.polar.html#matplotlib.pyplot.polar) | Make a polar plot.                                           |
| [`psd`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.psd.html#matplotlib.pyplot.psd) | Plot the power spectral density.                             |
| [`quiver`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html#matplotlib.pyplot.quiver) | Plot a 2D field of arrows.                                   |
| [`quiverkey`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiverkey.html#matplotlib.pyplot.quiverkey) | Add a key to a quiver plot.                                  |
| [`rc`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.rc.html#matplotlib.pyplot.rc) | Set the current rc params.                                   |
| [`rc_context`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.rc_context.html#matplotlib.pyplot.rc_context) | Return a context manager for managing rc settings.           |
| [`rcdefaults`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.rcdefaults.html#matplotlib.pyplot.rcdefaults) | Restore the rc params from Matplotlib's internal default style. |
| [`rgrids`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.rgrids.html#matplotlib.pyplot.rgrids) | Get or set the radial gridlines on the current polar plot.   |
| [`savefig`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html#matplotlib.pyplot.savefig) | Save the current figure.                                     |
| [`sca`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.sca.html#matplotlib.pyplot.sca) | Set the current Axes instance to *ax*.                       |
| [`scatter`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter) | A scatter plot of *y* vs *x* with varying marker size and/or color. |
| [`sci`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.sci.html#matplotlib.pyplot.sci) | Set the current image.                                       |
| [`semilogx`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.semilogx.html#matplotlib.pyplot.semilogx) | Make a plot with log scaling on the x axis.                  |
| [`semilogy`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.semilogy.html#matplotlib.pyplot.semilogy) | Make a plot with log scaling on the y axis.                  |
| [`set_cmap`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.set_cmap.html#matplotlib.pyplot.set_cmap) | Set the default colormap.                                    |
| [`setp`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.setp.html#matplotlib.pyplot.setp) | Set a property on an artist object.                          |
| [`show`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.show.html#matplotlib.pyplot.show) | Display a figure.                                            |
| [`specgram`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.specgram.html#matplotlib.pyplot.specgram) | Plot a spectrogram.                                          |
| [`spy`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.spy.html#matplotlib.pyplot.spy) | Plot the sparsity pattern of a 2D array.                     |
| [`stackplot`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.stackplot.html#matplotlib.pyplot.stackplot) | Draw a stacked area plot.                                    |
| [`stem`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.stem.html#matplotlib.pyplot.stem) | Create a stem plot.                                          |
| [`step`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.step.html#matplotlib.pyplot.step) | Make a step plot.                                            |
| [`streamplot`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.streamplot.html#matplotlib.pyplot.streamplot) | Draw streamlines of a vector flow.                           |
| [`subplot`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot) | Add a subplot to the current figure.                         |
| [`subplot2grid`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot2grid.html#matplotlib.pyplot.subplot2grid) | Create an axis at specific location inside a regular grid.   |
| [`subplot_tool`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot_tool.html#matplotlib.pyplot.subplot_tool) | Launch a subplot tool window for a figure.                   |
| [`subplots`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots) | Create a figure and a set of subplots.                       |
| [`subplots_adjust`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots_adjust.html#matplotlib.pyplot.subplots_adjust) | Tune the subplot layout.                                     |
| [`suptitle`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.suptitle.html#matplotlib.pyplot.suptitle) | Add a centered title to the figure.                          |
| [`switch_backend`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.switch_backend.html#matplotlib.pyplot.switch_backend) | Close all open figures and set the Matplotlib backend.       |
| [`table`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.table.html#matplotlib.pyplot.table) | Add a table to an [`Axes`](https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes). |
| [`text`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.text.html#matplotlib.pyplot.text) | Add text to the axes.                                        |
| [`thetagrids`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.thetagrids.html#matplotlib.pyplot.thetagrids) | Get or set the theta gridlines on the current polar plot.    |
| [`tick_params`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.tick_params.html#matplotlib.pyplot.tick_params) | Change the appearance of ticks, tick labels, and gridlines.  |
| [`ticklabel_format`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ticklabel_format.html#matplotlib.pyplot.ticklabel_format) | Change the [`ScalarFormatter`](https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.ScalarFormatter) used by default for linear axes. |
| [`tight_layout`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.tight_layout.html#matplotlib.pyplot.tight_layout) | Automatically adjust subplot parameters to give specified padding. |
| [`title`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.title.html#matplotlib.pyplot.title) | Set a title for the axes.                                    |
| [`tricontour`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.tricontour.html#matplotlib.pyplot.tricontour) | Draw contours on an unstructured triangular grid.            |
| [`tricontourf`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.tricontourf.html#matplotlib.pyplot.tricontourf) | Draw contours on an unstructured triangular grid.            |
| [`tripcolor`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.tripcolor.html#matplotlib.pyplot.tripcolor) | Create a pseudocolor plot of an unstructured triangular grid. |
| [`triplot`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.triplot.html#matplotlib.pyplot.triplot) | Draw a unstructured triangular grid as lines and/or markers. |
| [`twinx`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.twinx.html#matplotlib.pyplot.twinx) | Make and return a second axes that shares the *x*-axis.      |
| [`twiny`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.twiny.html#matplotlib.pyplot.twiny) | Make and return a second axes that shares the *y*-axis.      |
| [`uninstall_repl_displayhook`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.uninstall_repl_displayhook.html#matplotlib.pyplot.uninstall_repl_displayhook) | Uninstall the matplotlib display hook.                       |
| [`violinplot`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.violinplot.html#matplotlib.pyplot.violinplot) | Make a violin plot.                                          |
| [`vlines`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.vlines.html#matplotlib.pyplot.vlines) | Plot vertical lines.                                         |
| [`xcorr`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xcorr.html#matplotlib.pyplot.xcorr) | Plot the cross correlation between *x* and *y*.              |
| [`xkcd`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xkcd.html#matplotlib.pyplot.xkcd) | Turn on [xkcd](https://xkcd.com/) sketch-style drawing mode. |
| [`xlabel`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xlabel.html#matplotlib.pyplot.xlabel) | Set the label for the x-axis.                                |
| [`xlim`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xlim.html#matplotlib.pyplot.xlim) | Get or set the x limits of the current axes.                 |
| [`xscale`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xscale.html#matplotlib.pyplot.xscale) | Set the x-axis scale.                                        |
| [`xticks`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.xticks.html#matplotlib.pyplot.xticks) | Get or set the current tick locations and labels of the x-axis. |
| [`ylabel`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ylabel.html#matplotlib.pyplot.ylabel) | Set the label for the y-axis.                                |
| [`ylim`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ylim.html#matplotlib.pyplot.ylim) | Get or set the y-limits of the current axes.                 |
| [`yscale`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.yscale.html#matplotlib.pyplot.yscale) | Set the y-axis scale.                                        |
| [`yticks`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.yticks.html#matplotlib.pyplot.yticks) | Get or set the current tick locations and labels of the y-axis. |



## matplotlib.patches

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`Arc`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Arc.html#matplotlib.patches.Arc)(xy, width, height[, angle, theta1, theta2]) | An elliptical arc, i.e.                                      |
| [`Arrow`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Arrow.html#matplotlib.patches.Arrow)(x, y, dx, dy[, width]) | An arrow patch.                                              |
| [`ArrowStyle`](https://matplotlib.org/api/_as_gen/matplotlib.patches.ArrowStyle.html#matplotlib.patches.ArrowStyle) | [`ArrowStyle`](https://matplotlib.org/api/_as_gen/matplotlib.patches.ArrowStyle.html#matplotlib.patches.ArrowStyle) is a container class which defines several arrowstyle classes, which is used to create an arrow path along a given path. |
| [`BoxStyle`](https://matplotlib.org/api/_as_gen/matplotlib.patches.BoxStyle.html#matplotlib.patches.BoxStyle) | [`BoxStyle`](https://matplotlib.org/api/_as_gen/matplotlib.patches.BoxStyle.html#matplotlib.patches.BoxStyle) is a container class which defines several boxstyle classes, which are used for [`FancyBboxPatch`](https://matplotlib.org/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch). |
| [`Circle`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Circle.html#matplotlib.patches.Circle)(xy[, radius]) | A circle patch.                                              |
| [`CirclePolygon`](https://matplotlib.org/api/_as_gen/matplotlib.patches.CirclePolygon.html#matplotlib.patches.CirclePolygon)(xy[, radius, resolution]) | A polygon-approximation of a circle patch.                   |
| [`ConnectionPatch`](https://matplotlib.org/api/_as_gen/matplotlib.patches.ConnectionPatch.html#matplotlib.patches.ConnectionPatch)(xyA, xyB, coordsA[, ...]) | A [`ConnectionPatch`](https://matplotlib.org/api/_as_gen/matplotlib.patches.ConnectionPatch.html#matplotlib.patches.ConnectionPatch) class is to make connecting lines between two points (possibly in different axes). |
| [`ConnectionStyle`](https://matplotlib.org/api/_as_gen/matplotlib.patches.ConnectionStyle.html#matplotlib.patches.ConnectionStyle) | [`ConnectionStyle`](https://matplotlib.org/api/_as_gen/matplotlib.patches.ConnectionStyle.html#matplotlib.patches.ConnectionStyle) is a container class which defines several connectionstyle classes, which is used to create a path between two points. |
| [`Ellipse`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Ellipse.html#matplotlib.patches.Ellipse)(xy, width, height[, angle]) | A scale-free ellipse.                                        |
| [`FancyArrow`](https://matplotlib.org/api/_as_gen/matplotlib.patches.FancyArrow.html#matplotlib.patches.FancyArrow)(x, y, dx, dy[, width, ...]) | Like Arrow, but lets you set head width and head height independently. |
| [`FancyArrowPatch`](https://matplotlib.org/api/_as_gen/matplotlib.patches.FancyArrowPatch.html#matplotlib.patches.FancyArrowPatch)([posA, posB, path, ...]) | A fancy arrow patch.                                         |
| [`FancyBboxPatch`](https://matplotlib.org/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch)(xy, width, height[, ...]) | A fancy box around a rectangle with lower left at *xy* = (*x*, *y*) with specified width and height. |
| [`Patch`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch)([edgecolor, facecolor, color, ...]) | A patch is a 2D artist with a face color and an edge color.  |
| [`PathPatch`](https://matplotlib.org/api/_as_gen/matplotlib.patches.PathPatch.html#matplotlib.patches.PathPatch)(path, **kwargs) | A general polycurve path patch.                              |
| [`Polygon`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Polygon.html#matplotlib.patches.Polygon)(xy[, closed]) | A general polygon patch.                                     |
| [`Rectangle`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html#matplotlib.patches.Rectangle)(xy, width, height[, angle]) | A rectangle with lower left at *xy* = (*x*, *y*) with specified *width*, *height* and rotation *angle*. |
| [`RegularPolygon`](https://matplotlib.org/api/_as_gen/matplotlib.patches.RegularPolygon.html#matplotlib.patches.RegularPolygon)(xy, numVertices[, radius, ...]) | A regular polygon patch.                                     |
| [`Shadow`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Shadow.html#matplotlib.patches.Shadow)(patch, ox, oy[, props]) | Create a shadow of the given *patch* offset by *ox*, *oy*.   |
| [`Wedge`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Wedge.html#matplotlib.patches.Wedge)(center, r, theta1, theta2[, width]) | Wedge shaped patch.                                          |
| [`YAArrow`](https://matplotlib.org/api/_as_gen/matplotlib.patches.YAArrow.html#matplotlib.patches.YAArrow)(**kwargs) | [*Deprecated*] Yet another arrow class.                      |



### Patch类

*class* `matplotlib.patches.Patch`(*edgecolor=None*, *facecolor=None*, *color=None*, *linewidth=None*, *linestyle=None*, *antialiased=None*, *hatch=None*, *fill=True*, *capstyle=None*, joinstyle=None, ***kwargs*)

| 参数        | 描述     |
| ----------- | -------- |
| edgecolor   | 边框颜色 |
| facecolor   | 填充颜色 |
| color       | 所有颜色 |
| linewidth   | 线条宽度 |
| linestyle   | 线条样式 |
| antialiased | 防伪     |
| hatch       | 阴影     |
| fill        | 填充     |
| capstyle    |          |
| joinstyle   |          |
| **kwargs    |          |



| **kwargs可用参数                                             | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`agg_filter`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_agg_filter.html#matplotlib.artist.Artist.set_agg_filter) | a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array |
| [`alpha`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_alpha) | float or None                                                |
| [`animated`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_animated.html#matplotlib.artist.Artist.set_animated) | bool                                                         |
| [`antialiased`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_antialiased) or aa | unknown                                                      |
| [`capstyle`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_capstyle) | {'butt', 'round', 'projecting'}                              |
| [`clip_box`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_clip_box.html#matplotlib.artist.Artist.set_clip_box) | Bbox                                                         |
| [`clip_on`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_clip_on.html#matplotlib.artist.Artist.set_clip_on) | bool                                                         |
| [`clip_path`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_clip_path.html#matplotlib.artist.Artist.set_clip_path) | [([`Path`](https://matplotlib.org/api/path_api.html#matplotlib.path.Path), [`Transform`](https://matplotlib.org/api/transformations.html#matplotlib.transforms.Transform)) | [`Patch`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch) \| None] |
| [`color`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_color) | color                                                        |
| [`contains`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_contains.html#matplotlib.artist.Artist.set_contains) | callable                                                     |
| [`edgecolor`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_edgecolor) or ec | color or None or 'auto'                                      |
| [`facecolor`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_facecolor) or fc | color or None                                                |
| [`figure`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_figure.html#matplotlib.artist.Artist.set_figure) | [`Figure`](https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure) |
| [`fill`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_fill) | bool                                                         |
| [`gid`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_gid.html#matplotlib.artist.Artist.set_gid) | str                                                          |
| [`hatch`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_hatch) | {'/', '\', '\|', '-', '+', 'x', 'o', 'O', '.', '*'}          |
| [`in_layout`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_in_layout.html#matplotlib.artist.Artist.set_in_layout) | bool                                                         |
| [`joinstyle`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_joinstyle) | {'miter', 'round', 'bevel'}                                  |
| [`label`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_label.html#matplotlib.artist.Artist.set_label) | object                                                       |
| [`linestyle`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_linestyle) or ls | {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}        |
| [`linewidth`](https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_linewidth) or lw | float or None                                                |
| [`path_effects`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_path_effects.html#matplotlib.artist.Artist.set_path_effects) | [`AbstractPathEffect`](https://matplotlib.org/api/patheffects_api.html#matplotlib.patheffects.AbstractPathEffect) |
| [`picker`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_picker.html#matplotlib.artist.Artist.set_picker) | None or bool or float or callable                            |
| [`rasterized`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_rasterized.html#matplotlib.artist.Artist.set_rasterized) | bool or None                                                 |
| [`sketch_params`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_sketch_params.html#matplotlib.artist.Artist.set_sketch_params) | (scale: float, length: float, randomness: float)             |
| [`snap`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_snap.html#matplotlib.artist.Artist.set_snap) | bool or None                                                 |
| [`transform`](https://matplotlib.org/api/_as_gen/matplotlib.artist.Artist.set_transform.html#matplotlib.artist.Artist.set_transform) | [`Transform`](https://matplotlib.org/api/transformations.html#matplotlib.transforms.Transform) |
| `url`                                                        | str                                                          |
| `visible`                                                    | bool                                                         |
| `zorder`                                                     | float                                                        |







## demo

### 画矩形

```python
rect = plt.Rectangle((0.1,0.1),0.5,0.3,fill=False,edgecolor='r')
plt.figure()
plt.gca().add_patch(rect)
plt.show()
```

