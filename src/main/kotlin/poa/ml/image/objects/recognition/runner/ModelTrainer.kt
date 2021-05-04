package poa.ml.image.objects.recognition.runner

import kotlinx.coroutines.runBlocking
import poa.ml.image.objects.recognition.*
import poa.ml.image.objects.recognition.labeler.FixedImageSampleLabeler
import poa.ml.image.objects.recognition.labeler.ManualImageSampleLabeler
import poa.ml.image.objects.recognition.model.LabeledTrainingSet
import java.util.concurrent.atomic.AtomicInteger

class ModelTrainer {

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

    fun train(dir: String, fileToSave: String) {
        val labeledTrainingSet = LabeledTrainingSet()
        val counter = AtomicInteger()
        walkFileTree(dir) { img ->
            runBlocking {

                val resizedImg = img.resized(targetHeight = 600)

                val (goodChunk, badChunks) = imageCutter.cut(resizedImg, "Image №${counter.incrementAndGet()}")

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
                        val samples = smallStepSampleCollector.collect(goodChunk.image, it)
                        imageSampleLabeler.label(samples, true)
                            .apply { labeledTrainingSet.add(this) }
                    }
            }
        }
        labeledTrainingSet.save(fileToSave)
        println("===Training set with ${labeledTrainingSet.size()} rows saved.")
        println("===Path: $fileToSave")
        println("===Positive labels percentage: ${labeledTrainingSet.positiveLabelsPercentage()}%")
        println("")
    }

}
