import pygame
import time

WIDTH = 800
HEIGHT = 480


def print_events():
    for event in pygame.event.get():
        # print(pygame.event.event_name(event.type))
        print_mouse_events(event)
        
        check_for_quit(event)


def print_mouse_events(event) -> None:
    if event.type == pygame.MOUSEBUTTONDOWN:
        print("Mouse down at", event.pos)

    elif event.type == pygame.MOUSEMOTION:
        if pygame.mouse.get_pressed()[0]:
            print("Mouse dragging at", event.pos)

    elif event.type == pygame.MOUSEBUTTONUP:
        print("Mouse up at", event.pos)


def check_for_quit(event):
    if event.type == pygame.QUIT:
        pygame.quit()
        raise SystemExit

    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        pygame.quit()
        raise SystemExit


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    pygame.display.set_caption("Touch Test")

    while True:
        print_events()
