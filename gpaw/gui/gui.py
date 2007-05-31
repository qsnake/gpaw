#!/usr/bin/env python
"""
husk:
Exit*2?
close button
DFT
Atoms: cell, periodicity
ADOS
units?
grey-out stuff after one second: vmd, rasmol, ...
Show with ...
rasmol: set same rotation as g2
Graphs: save, Python, 3D
start from python (interactive mode?)
ascii-art option (colored)
option -o (output) and -f (force overwrite)
surfacebuilder
screen-dump
icon
g2-community-server
translate option: record all translations, and check for missing translations.
"""

import os
import sys

import numpy as npy

import gtk
from gpaw.gui.view import View
from gpaw.gui.status import Status
from gpaw.gui.languages import translate as _
from gpaw.gui.read import read_from_files
from gpaw.gui.write import write_to_file
from gpaw.gui.widgets import pack, help, Help


ui_info = """\
<ui>
  <menubar name='MenuBar'>
    <menu action='FileMenu'>
      <menuitem action='Open'/>
      <menuitem action='New'/>
      <menuitem action='Save'/>
      <separator/>
      <menuitem action='Quit'/>
    </menu>
    <menu action='EditMenu'>
      <menuitem action='Select'/>
      <menuitem action='SelectAll'/>
      <menuitem action='Invert'/>
      <menuitem action='SelectConstrained'/>
      <separator/>
      <menuitem action='First'/>
      <menuitem action='Previous'/>
      <menuitem action='Next'/>
      <menuitem action='Last'/>
    </menu>
    <menu action='ViewMenu'>
      <menuitem action='ShowUnitCell'/>
      <menuitem action='ShowAxes'/>
      <separator/>
      <menuitem action='Repeat'/>
      <menuitem action='Focus'/>
      <menuitem action='ZoomIn'/>
      <menuitem action='ZoomOut'/>
      <menuitem action='Settings'/>
        <menuitem action='VMD'/>
        <menuitem action='RasMol'/>
        <menuitem action='XMakeMol'/>
    </menu>
    <menu action='ToolsMenu'>
      <menuitem action='Graphs'/>
      <menuitem action='Movie'/>
      <menuitem action='Modify'/>
      <menuitem action='Constraints'/>
      <menuitem action='DFT'/>
      <menuitem action='NEB'/>
      <menuitem action='DOS'/>
      <menuitem action='Wannier'/>
    </menu>
    <menu action='HelpMenu'>
      <menuitem action='About'/>
      <menuitem action='Wiki'/>
      <menuitem action='Debug'/>
    </menu>
  </menubar>
</ui>"""

class GUI(View, Status):
    def __init__(self, atoms, rotations, show_unit_cell):
        self.atoms = atoms
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        #self.window.set_icon(gtk.gdk.pixbuf_new_from_file('guiase.png'))
        self.window.set_position(gtk.WIN_POS_CENTER)
        #self.window.connect("destroy", lambda w: gtk.main_quit())
        self.window.connect('delete_event', self.exit)
        self.window.set_title('g2')
        vbox = gtk.VBox()
        self.window.add(vbox)
        self.set_tip = gtk.Tooltips().set_tip

        actions = gtk.ActionGroup("Actions")
        actions.add_actions([
            ('FileMenu', None, '_File'),
            ('EditMenu', None, '_Edit'),
            ('ViewMenu', None, '_View'  ),
            ('ToolsMenu', None, '_Tools'),
            ('HelpMenu', None, '_Help'),
            ('Open', gtk.STOCK_OPEN, '_Open', '<control>O',
             'Create a new file',
             self.open),
            ('New', gtk.STOCK_NEW, '_New', '<control>N',
             'New g2 window',
             lambda widget: os.system('g2 &')),
            ("Save", gtk.STOCK_SAVE, "_Save", "<control>S",
             "Save current file",
             self.save),
            ("Quit", gtk.STOCK_QUIT, "_Quit", "<control>Q",
             "Quit",
             self.exit),
            ('Select', None, '_Select ...', None,
             '',
             self.xxx),
            ('SelectAll', None, 'Select _all', None,
             '',
             self.select_all),
            ('Invert', None, '_Invert selection', None,
             '',
             self.invert_selection),
            ('SelectConstrained', None, 'Select _constrained atoms', None,
             '',
             self.select_constrained_atoms),
            ('First', gtk.STOCK_GOTO_FIRST, '_First image', 'Home',
             '',
             self.step),
            ('Previous', gtk.STOCK_GO_BACK, '_Previous image', 'Page_Up',
             '',
             self.step),
            ('Next', gtk.STOCK_GO_FORWARD, '_Next image', 'Page_Down',
             '',
             self.step),
            ('Last', gtk.STOCK_GOTO_LAST, '_Last image', 'End',
             '',
             self.step),
            ('Repeat', None, 'Repeat ...', None,
             '',
             self.repeat_window),
            ('Focus', gtk.STOCK_ZOOM_FIT, 'Focus', 'F',
             '',
             self.focus),
            ('ZoomIn', gtk.STOCK_ZOOM_IN, 'Zoom in', 'plus',
             '',
             self.zoom),
            ('ZoomOut', gtk.STOCK_ZOOM_OUT, 'Zoom out', 'minus',
             '',
             self.zoom),
            ('Settings', gtk.STOCK_PREFERENCES, 'Settings ...', None,
             '',
             self.xxx),
            ('VMD', None, 'VMD', None,
             '',
             self.external_viewer),
            ('RasMol', None, 'RasMol', None,
             '',
             self.external_viewer),
            ('XMakeMol', None, 'xmakemol', None,
             '',
             self.external_viewer),
            ('Graphs', None, 'Graphs ...', None,
             '',
             self.plot_graphs),
            ('Movie', None, 'Movie ...', None,
             '',
             self.movie),
            ('Modify', None, 'Modify ...', None,
             '',
             self.execute),
            ('Constraints', None, 'Constraints ...', None,
             '',
             self.constraints_window),
            ('DFT', None, 'DFT ...', None,
             '',
             self.dft_window),
            ('NEB', None, 'NE_B', None,
             '',
             self.NEB),
            ('DOS', None, 'DOS ...', None,
             '',
             self.xxx),
            ('Wannier', None, 'Wannier ...', None,
             '',
             self.xxx),
            ('About', None, '_About', None,
             None,
             self.about),
            ('Wiki', gtk.STOCK_HELP, 'Wiki ...', None, None, wiki),
            ('Debug', None, 'Debug ...', None, None, self.debug)])
        actions.add_toggle_actions([
            ("ShowUnitCell", None, "Show _unit cell", "<control>U",
             "Bold",
             self.toggle_show_unit_cell,
             show_unit_cell > 0),
            ("ShowAxes", None, "Show _axes", "<control>A",
             "Bold",
             self.toggle_show_axes,
             True)])
        self.ui = ui = gtk.UIManager()
        ui.insert_action_group(actions, 0)
        self.window.add_accel_group(ui.get_accel_group())

        try:
            mergeid = ui.add_ui_from_string(ui_info)
        except gobject.GError, msg:
            print "building menus failed: %s" % msg

        vbox.pack_start(ui.get_widget("/MenuBar"), False, False, 0)
        #ui.get_widget("/MenuBar/FileMenu").set_tooltips(True)
        #gtk.Tooltips().enable()
        
        View.__init__(self, vbox, rotations)
        Status.__init__(self, vbox)
        vbox.show()
        #self.window.set_events(gtk.gdk.BUTTON_PRESS_MASK)
        self.window.connect("key-press-event", self.scroll)
        self.window.show()
        self.graphs = []
        self.movie_window = None
        
    def run(self, expr):
        self.set_colors()
        self.set_coordinates(self.atoms.nframes - 1, focus=True)

        if self.atoms.nframes > 1:
            self.movie()

        if expr is None:
            expr = 'i, e - E[-1]'
            
        if expr is not None and self.atoms.nframes > 1:
            self.plot_graphs(expr=expr)

        gtk.main()

    def step(self, action):
        d = {'First': -10000000,
             'Previous': -1,
             'Next': 1,
             'Last': 10000000}[action.get_name()]
        i = max(0, min(self.atoms.nframes - 1, self.frame + d))
        self.set_frame(i)
        if self.movie_window is not None:
            self.movie_window.frame_number.value = i
            
    def zoom(self, action):
        x = {'ZoomIn': 1.2, 'ZoomOut':1 / 1.2}[action.get_name()]
        self.scale *= x
        center = (0.5 * self.width, 0.5 * self.height, 0)
        self.offset = x * (self.offset + center) - center
        self.draw()

    def scroll(self, window, event):
        dxdy = {gtk.keysyms.Up:    ( 0, -1),
                gtk.keysyms.Down:  ( 0, +1),
                gtk.keysyms.Right: (+1,  0),
                gtk.keysyms.Left:  (-1,  0)}.get(event.keyval, None)
        if dxdy is None:
            return
        dx, dy = dxdy
        d = self.scale * 0.1
        self.offset -= (dx * d, dy * d, 0)
        self.draw()
                
    def debug(self, x):
        from gpaw.gui.debug import Debug
        Debug(self)

    def execute(self, widget=None):
        from gpaw.gui.execute import Execute
        Execute(self)
        
    def constraints_window(self, widget=None):
        from gpaw.gui.constraints import Constraints
        Constraints(self)

    def dft_window(self, widget=None):
        from gpaw.gui.dft import DFT
        DFT(self)

    def select_all(self, widget):
        self.atoms.selected[:] = True
        self.draw()
        
    def invert_selection(self, widget):
        self.atoms.selected[:] = ~self.atoms.selected
        self.draw()
        
    def select_constrained_atoms(self, widget):
        self.atoms.selected[:] = ~self.atoms.dynamic
        self.draw()
        
    def movie(self, widget=None):
        from gpaw.gui.movie import Movie
        self.movie_window = Movie(self)
        
    def plot_graphs(self, x=None, expr=None):
        from gpaw.gui.graphs import Graphs
        g = Graphs(self)
        if expr is not None:
            g.plot(expr=expr)
        
    def NEB(self, action):
        from gpaw.gui.neb import NudgedElasticBand
        NudgedElasticBand(self.atoms)
        
    def open(self, button=None, filenames=None, slice=':'):
        if filenames == None:
            chooser = gtk.FileChooserDialog(
                _('Open ...'), None, gtk.FILE_CHOOSER_ACTION_OPEN,
                (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                 gtk.STOCK_OPEN, gtk.RESPONSE_OK))
            ok = chooser.run()
            if ok == gtk.RESPONSE_OK:
                filenames = [chooser.get_filename()]
            chooser.destroy()

            if not ok:
                return
            
        cell, periodic, Z, dft, RR, EE, FF = read_from_files(filenames, slice)
        self.atoms.set_atoms(cell, periodic, Z, RR, EE, FF)
        self.atoms.dft = dft
        self.set_colors()
        self.set_coordinates(self.atoms.nframes - 1, focus=True)

    def save(self, action):
        chooser = gtk.FileChooserDialog(
            _('Save ...'), None, gtk.FILE_CHOOSER_ACTION_SAVE,
            (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
             gtk.STOCK_SAVE, gtk.RESPONSE_OK))

        combo = gtk.combo_box_new_text()
        types = []
        for name, suffix in [('Automatic', None),
                             ('XYZ file', 'xyz'),
                             ('ASE trajectory', 'traj'),
                             ('PDB file', 'pdb'),
                             ('Python script', 'py'),
                             ('VNL file', 'vnl'),
                             ('Portable Network Graphics', 'png'),
                             ('Encapsulated PostScript', 'eps')]:
            if suffix is None:
                name = _(name)
            else:
                name = '%s (%s)' % (_(name), suffix)
            types.append(suffix)
            combo.append_text(name)

        combo.set_active(0)

        pack(chooser.vbox, combo)

        if self.atoms.nframes > 1:
            button = gtk.CheckButton('Save current frame only (#%d)' %
                                     self.frame)
            pack(chooser.vbox, button)
            entry = pack(chooser.vbox, [gtk.Label(_('Slice: ')),
                                        gtk.Entry(10),
                                        help('Help for slice ...')])[1]
            entry.set_text('0:%d' % self.atoms.nframes)

        while True:
            if chooser.run() == gtk.RESPONSE_CANCEL:
                chooser.destroy()
                return
            
            filename = chooser.get_filename()

            i = combo.get_active()
            if i == 0:
                suffix = filename.split('.')[-1]
                if suffix not in types:
                   self.xxx(message1='Unknown output format!',
                            message2='Use one of: ' + ', '.join(types[1:]))
                   continue
            else:
                suffix = types[i]
                
            if suffix in ['pdb', 'vnl']:
                self.xxx()
                continue
                
            if self.atoms.nframes == 1 or button.get_active():
                frames = [self.frame]
            else:
                frames = range(self.atoms.nframes)
                slice = entry.get_text()
                frames = eval('frames[%s]' % slice)
                if isinstance(frames, int):
                    frames = [frames]
                if len(frames) == 0:
                    self.xxx(message1='Empty selection!',
                             message2='Press Help-button for help.')
                    continue

            # Does filename exist?
            
            break
        
        chooser.destroy()

        write_to_file(filename, self.atoms, suffix, frames, self)
        
    def exit(self, button, event=None):
        gtk.main_quit()
        return True

    def xxx(self, x=None,
            message1='Not implemented!',
            message2='do you really need it?'):
        dialog = gtk.MessageDialog(flags=gtk.DIALOG_MODAL,
                                   type=gtk.MESSAGE_WARNING,
                                   buttons=gtk.BUTTONS_CLOSE,
                                   message_format=_(message1))
        try:
            dialog.format_secondary_text(_(message2))
        except AttributeError:
            print _(message2)
        dialog.connect('response', lambda x, y: dialog.destroy())
        dialog.show()
        
    def about(self, action):
        try:
            dialog = gtk.AboutDialog()
        except AttributeError:
            self.xxx()
        else:
            dialog.run()

def wiki(widget):
    import webbrowser
    webbrowser.open('https://wiki.fysik.dtu.dk/gpaw/G2')
