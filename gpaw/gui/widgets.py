import gtk

from gpaw.gui.languages import translate as _


class Number(gtk.SpinButton):
    def __init__(self, value=0,
                 lower=0, upper=10000,
                 step_incr=1, page_incr=10,
                 climb_rate=0.5, digits=0):
        self.adj = gtk.Adjustment(value, lower, upper, step_incr, page_incr, 0)
        gtk.SpinButton.__init__(self, self.adj, climb_rate, digits)

    def connect(self, *args):
        return self.adj.connect(*args)


class Menu:
    def __init__(self, menubar, name, items):
        self.items = {}
        menu = gtk.Menu()
        for data in items:
            text = data[0]
            callback = data[1]
            args = data[2:]
            menuitem = gtk.MenuItem(_(text))
            menu.append(menuitem)
            menuitem.connect('activate', callback, *args)
            menuitem.show()
            self.items[text] = menuitem
        menuitem = gtk.MenuItem(_(name))
        menubar.append(menuitem)
        menuitem.set_submenu(menu)
        menuitem.show()


class Help(gtk.Window):
    def __init__(self, text):
        gtk.Window.__init__(self)
        vbox = gtk.VBox()
        self.add(vbox)
        label = pack(vbox, gtk.Label())
        label.set_line_wrap(True)
        text = _(text).replace('<c>', '<span foreground="blue">')
        text = text.replace('</c>', '</span>')
        label.set_markup(text)
        close = pack(vbox, gtk.Button(_('Close')))
        close.connect('clicked', lambda widget: self.destroy())
        self.show_all()

def help(text):
    button = gtk.Button(_('Help'))
    button.connect('clicked', lambda widget, text=text: Help(text))
    return button


class Window(gtk.Window):
    def __init__(self, gui):
        self.gui = gui
        gtk.Window.__init__(self)
        self.set_title(_('Constraints'))
        vbox = gtk.VBox()
        b = pack(vbox, [gtk.Button(_('Constrain')),
                        gtk.Label(_(' selected atoms'))])[0]
        b.connect('clicked', self.selected)
        b = pack(vbox, [gtk.Button(_('Constrain')),
                        gtk.Label(_(' immobile atoms:'))])[0]
        b.connect('clicked', self.immobile)
        b = pack(vbox, gtk.Button('Clear constraint'))
        b.connect('clicked', self.clear)
        close = pack(vbox, gtk.Button(_('Close')))
        close.connect('clicked', lambda widget: self.destroy())
        self.add(vbox)
        vbox.show()
        self.show()

def pack(vbox, widgets, end=False):
    if not isinstance(widgets, list):
        widgets.show()
        vbox.pack_start(widgets, 0, 0)
        return widgets
    hbox = gtk.HBox(0, 0)
    hbox.show()
    vbox.pack_start(hbox, 0, 0)
    for widget in widgets:
        if type(widget) is gtk.Entry:
            widget.set_size_request(widget.get_max_length() * 9, 24)
        widget.show()
        if end and widget is widgets[-1]:
            hbox.pack_end(widget, 0, 0)
        else:
            hbox.pack_start(widget, 0, 0)
    return widgets
