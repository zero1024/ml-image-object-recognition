package poa.ml.image.objects.recognition

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import smile.classification.svm
import smile.math.kernel.GaussianKernel
import smile.math.matrix.Matrix
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

class Tester {

    private val imageSamplesCollector = ImageSamplesCollector(
        pxlStep = 60,
        slideSize = 60
    )
    private val imageSampleLabeler = ImageSampleLabeler()


    @Test
    internal fun testAllTheThings() {

        runBlocking {
            val testImage = testImage().resized(targetHeight = 600)
            val samples = imageSamplesCollector.collect(testImage)
            val labeledSamples = imageSampleLabeler.label(samples, testImage)
            val (X, y) = toTrainingSet(labeledSamples)

            val center = X.colMeans()
            val scale = X.colSds()

            val xScaled = X.scale(center, scale).toArray()

//            val classifier = mlp(xScaled, y, arrayOf(Layer.sigmoid(10), Layer.mle(1, OutputFunction.SIGMOID)))
//            val classifier = logit(xScaled, y)
            val classifier = svm(xScaled, y, GaussianKernel(4.0), 1.0)

            for (sample in samples) {
                val image = sample.image
                val array = Matrix(arrayOf(toDoubleArray(image))).scale(center, scale).row(0)
                val label = classifier.predict(array)
                if (label == 1) {
                    showImage(image)
                }
            }
            Thread.sleep(100000000)

        }
    }
}


private fun testImage(): BufferedImage {
    return ImageIO.read(File("/Users/oleg1024/Downloads/iriska.jpg"))
}
