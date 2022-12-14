import io
import pygame as pg
from urllib.request import urlopen

scale = 0.05

pg.init()
screen = pg.display.set_mode((640,480),0,32)
pg.display.set_caption("Hello pg")
screen.fill((0,0,0))
game_over = False
clock = pg.time.Clock()

url2b = 'https://raw.githubusercontent.com/zcemycl/practice-app/master/src/components/Random/2b.png'
image_str = urlopen(url2b).read()
# create a file object (stream)
image_file = io.BytesIO(image_str)
sp1 = pg.image.load(image_file)
w0,h0 = sp1.get_width(),sp1.get_height()
sp1 = pg.transform.scale(sp1,(int(w0*scale),int(h0*scale)))
w_,h_ = sp1.get_width(),sp1.get_height()

cors = []
while not game_over:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            game_over = True
        elif event.type == pg.MOUSEBUTTONUP:
            pos = pg.mouse.get_pos()
            cors.append(pos)

    pressed = pg.key.get_pressed()
    # print(pressed[pg.K_UP])
    screen.fill((0,0,0))

    prev = None
    for pos in cors:
        pg.draw.circle(
            screen, (122, 0, 122), pos, 7)
        if prev:
            pg.draw.line(screen, (0,255,0), pos, prev, 3)
        prev = pos
    
    screen.blit(sp1,(screen.get_width() - w_,
        screen.get_height() - h_))
    pg.display.update()
        

pg.quit()