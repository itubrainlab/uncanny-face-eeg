from psychopy import visual

from constants import outer_size, inner_size, FOREGROUND, BACKGROUND

def make_fix(win):
    outer_circle = visual.Circle(win, pos=(0,0), fillColor=FOREGROUND,
                                 units="deg", radius=outer_size, edges=64, colorSpace='rgb255')
    inner_circle = visual.Circle(win, pos=(0,0), fillColor=FOREGROUND,
                                 units="deg", radius=inner_size, edges=64, colorSpace='rgb255')
    rect1 = visual.Rect(win, pos=(0,0),fillColor=BACKGROUND, units="deg",
                        size=(inner_size*2, outer_size*2), colorSpace='rgb255')
    rect2 = visual.Rect(win, pos=(0,0), fillColor=BACKGROUND, units="deg",
                        size=(outer_size*2, inner_size*2), colorSpace='rgb255')

    fixList = [outer_circle, rect1, rect2, inner_circle ]

    return visual.BufferImageStim(win, stim=fixList,
                                  rect=(-1, 1, 1, -1),
                                  interpolate=False) # rect sizes can be tweaked..

if __name__ == '__main__':
    from psychopy import event
    win = visual.Window([800,600], allowGUI=True, monitor='testMonitor', units='pix',
                        color=BACKGROUND, colorSpace='rgb255')
    
    fix = make_fix(win)
    fix.draw()
    win.flip()
    event.waitKeys()