from renderer.threecircles import ThreeCircles
from renderer.rectangles import Rectangles

if __name__ == "__main__":
    scene = ThreeCircles(headless=False, radius=30)

    for frame in range(1000):
        scene.step()
        scene.draw()
