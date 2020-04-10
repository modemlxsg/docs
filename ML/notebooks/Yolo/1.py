from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)

generator = GeneratorFromStrings(
    ['Test1', 'Test2', 'Test3'],
    blur=2,
    random_blur=True
)

for img, lbl in generator:
    # Do something with the pillow images here.
    print(lbl)