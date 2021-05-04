package poa.ml.image.objects.recognition.runner

import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import poa.ml.image.objects.recognition.*
import poa.ml.image.objects.recognition.labeler.FixedImageSampleLabeler
import poa.ml.image.objects.recognition.labeler.ManualImageSampleLabeler
import poa.ml.image.objects.recognition.model.LabeledTrainingSet

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

    fun train(dir: String) {
        runBlocking {
            val labeledTrainingSet = LabeledTrainingSet()
            walkFileTree(dir) { img ->
                launch {

                    val resizedImg = img.resized(targetHeight = 600)

                    val (goodChunk, badChunks) = imageCutter.cut(resizedImg)

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
                }
            }
            println(labeledTrainingSet)
        }


    }

}