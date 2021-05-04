package poa.ml.image.objects.recognition

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import poa.ml.image.objects.recognition.labeler.FixedImageSampleLabeler
import poa.ml.image.objects.recognition.labeler.ManualImageSampleLabeler
import poa.ml.image.objects.recognition.model.LabeledTrainingSet
import smile.base.mlp.Layer
import smile.base.mlp.OutputFunction
import smile.classification.mlp
import java.awt.Toolkit
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

class Tester {

    private val samplesCollector = ImageSamplesCollector(
        pxlStep = 53,
        slideSize = 61
    )
    private val smallStepSampleCollector = ImageSamplesCollector(
        pxlStep = 20,
        slideSize = 61
    )
    private val manualImageSampleLabeler = ManualImageSampleLabeler()
    private val imageSampleLabeler = FixedImageSampleLabeler()
    private val imageCutter = ImageCutter()

    @Test
    internal fun testAllTheThings() {

        runBlocking {
            val labeledTrainingSet = LabeledTrainingSet()

            val (scaleK, testImage) = resize(testImage(), targetHeight = 600)
            val (goodChunk, badChunks) = imageCutter.cut(testImage)

            for (badChunk in badChunks) {
                val chunkSamples = smallStepSampleCollector.collect(badChunk.image)
                imageSampleLabeler.label(chunkSamples, false)
                    .apply { labeledTrainingSet.add(this) }
            }

            val chunkSamples = samplesCollector.collect(goodChunk.image)
            val goodSamples = manualImageSampleLabeler.label(chunkSamples, goodChunk.image)

            goodSamples
                .filter { (_, label) -> !label }
                .apply { labeledTrainingSet.add(this) }

            val positiveSamples = goodSamples.filter { (_, label) -> label }
            labeledTrainingSet.add(positiveSamples)

            positiveSamples
                .map { (sample, _) -> sample.toArea() }
                .let { AreaSums(it) }.forEach {
                    val samples = smallStepSampleCollector.collect(goodChunk.image, it)
                    imageSampleLabeler.label(samples, true)
                        .apply { labeledTrainingSet.add(this) }
                }

            val (X, y) = labeledTrainingSet.toMatrix()

            val center = X.colMeans()
            val scale = X.colSds()

            val xScaled = X.scale(center, scale).toArray()

            val classifier = mlp(xScaled, y, arrayOf(Layer.sigmoid(10), Layer.mle(1, OutputFunction.SIGMOID)))
//            val classifier = logit(xScaled, y)
//            val classifier = svm(xScaled, y, GaussianKernel(4.0), 1.0)

            val samples = samplesCollector.collect(testImage)
            val areaSums = samples
                .map { toDoubleArray(it.image).scale(center, scale) to it }
                .map { (array, sample) -> classifier.predict(array) to sample }
                .filter { (label, _) -> label == 1 }
                .map { (_, sample) -> sample.toArea() }
                .let { AreaSums(it) }

            val screenSize = Toolkit.getDefaultToolkit().screenSize.height.toDouble() * 0.8
            var (anotherScaleK, resultImage) = resize(testImage(), screenSize.toInt())
            for (area in areaSums) {
                resultImage = highlightArea(resultImage, area.scaled(scaleK / anotherScaleK))
            }
            showImage(resultImage)


            Thread.sleep(100000000)

        }
    }
}


private fun testImage(): BufferedImage {
    return ImageIO.read(File("/Users/oleg1024/Downloads/iriska.jpg"))
}
