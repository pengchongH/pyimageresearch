import imutils


def pyramid(image, scale=1.5, minSize=(30, 30)):  # minSize=(w, h)
    # yield the original image
    yield image

    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does meet the criterion
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield x, y, image[y: y + windowSize[1], x: x + windowSize[0]]
