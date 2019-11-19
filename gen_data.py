from experiments.npe.datagen import collect_data
from renderer.threecircles import ThreeCircles

import pickle

if __name__ == "__main__":
    train_x, train_y = collect_data(num_sequences=1000, sequence_length=500, seed=1337)
    val_x, val_y = collect_data(num_sequences=100, sequence_length=500, seed=5332)

    with open("data/threecircles_npe5.p", "wb") as f:
        pickle.dump({"train": (train_x, train_y), "val": (val_x, val_y)}, f)

    # scene = ThreeCircles(radius=30, headless=False)

    # for frame in range(100):
    #     scene.step()
    #     scene.draw()
