
__kernel void convolution2D(
    __global float * inputData, __global float * outputData, __constant float * maskData,
    int width, int height, int maskWidth,  int imageChannels){
    //@@ Insert code to implement matrix multiplication here

    int row = get_global_id(1);
    int col = get_global_id(0);

    int maskRadius = maskWidth / 2;

    if (row < height && col < width) {
        for (int k = 0; k < imageChannels; k++)
        {
            float accum = 0.0;
            for (int y = -maskRadius; y <= maskRadius; y++) {
                for (int x = -maskRadius; x <= maskRadius; x++) {
                    int xOffset = col + x;
                    int yOffset = row + y;
                    if (xOffset > -1 && xOffset < width && yOffset > -1 && yOffset < height) {
                        float imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + k];
                        float maskValue = maskData[(y+maskRadius)*maskWidth+x+maskRadius];
                        accum += imagePixel * maskValue;
                    }
                }
            }
            outputData[(row*width + col) * imageChannels + k] = clamp(accum, 0.0f, 1.0f);
        }
    }

    /**
    maskRadius := maskWidth/2 # this is integer division, so the result is 2
    for i from 0 to height do
    for j from 0 to width do
        for k from 0 to channels
        accum := 0
        for y from -maskRadius to maskRadius do
            for x from -maskRadius to maskRadius do
            xOffset := j + x
            yOffset := i + y
            if xOffset >= 0 && xOffset < width &&
                yOffset >= 0 && yOffset < height then
                imagePixel := I[(yOffset * width + xOffset) * channels + k]
                maskValue := K[(y+maskRadius)*maskWidth+x+maskRadius]
                accum += imagePixel * maskValue
            end
            end
        end
        # pixels are in the range of 0 to 1
        P[(i * width + j)*channels + k] = clamp(accum, 0, 1)
        end
    end
    end */
}