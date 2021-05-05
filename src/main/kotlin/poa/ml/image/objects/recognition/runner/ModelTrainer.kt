package poa.ml.image.objects.recognition.runner

import kotlinx.coroutines.runBlocking
import poa.ml.image.objects.recognition.*
import poa.ml.image.objects.recognition.labeler.FixedImageSampleLabeler
import poa.ml.image.objects.recognition.labeler.ManualImageSampleLabeler
import poa.ml.image.objects.recognition.model.LabeledTrainingSet
import java.util.concurrent.atomic.AtomicInteger

class ModelTrainer {

    private val samplesCollector = ImageSamplesCollector(
        pxlStep = 50,
        slideSize = 60
    )
    private val smallStepSampleCollector = ImageSamplesCollector(
        pxlStep = 30,
        slideSize = 60
    )
    private val verySmallStepSampleCollector = ImageSamplesCollector(
        pxlStep = 18,
        slideSize = 60
    )
    private val manualImageSampleLabeler = ManualImageSampleLabeler()
    private val imageSampleLabeler = FixedImageSampleLabeler()
    private val imageCutter = ImageCutter(minSize = 60)

    fun train(dir: String, fileToSave: String) {
        val labeledTrainingSet = LabeledTrainingSet()
        val counter = AtomicInteger()
        walkFileTree(dir) { img ->
            runBlocking {

                val resizedImg = img.resized(targetHeight = 400)

                val (goodChunk, badChunks) = imageCutter.cut(
                    resizedImg,
                    "Image №${counter.incrementAndGet()}. Memory used :${memoryUsed()}"
                )

                for (badChunk in badChunks) {
                    val negativeSamples = smallStepSampleCollector.collect(badChunk.image)
                    imageSampleLabeler.label(negativeSamples, false)
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
                        val samples = verySmallStepSampleCollector.collect(goodChunk.image, it)
                        imageSampleLabeler.label(samples, true)
                            .apply { labeledTrainingSet.add(this) }
                    }
            }
        }

        labeledTrainingSet.shuffle()

        val rowsNumber = labeledTrainingSet.size()
        val positiveLabelsPercentage = labeledTrainingSet.positiveLabelsPercentage()

        val centerAndScale = labeledTrainingSet.means() to labeledTrainingSet.sds()

        println("===Converting to matrix...")
        val (X, y) = labeledTrainingSet.toMatrix()
        println("===Done converting")

        println("===Scaling matrix...")
        val Xscaled = X.scale(centerAndScale.first, centerAndScale.second)
        println("===Done scaling")

        centerAndScale.saveToFile("$fileToSave.options")
        Xscaled.saveToFile("$fileToSave.X")
        y.saveToFile("$fileToSave.y")
        println("===Training set with $rowsNumber rows saved.")
        println("===Path: $fileToSave")
        println("===Positive labels percentage: $positiveLabelsPercentage%")
        println("")
    }

}
