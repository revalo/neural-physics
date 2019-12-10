from renderer.threecircles import ThreeCircles
from renderer.rectangles import Rectangles

if __name__ == "__main__":
    scene = Rectangles(headless=False, rand_height=False)

    for frame in range(1000):
        scene.step()
        scene.draw()
