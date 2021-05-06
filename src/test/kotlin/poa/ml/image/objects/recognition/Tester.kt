package poa.ml.image.objects.recognition

import org.junit.jupiter.api.Test
import poa.ml.image.objects.recognition.runner.ModelTrainer
import poa.ml.image.objects.recognition.runner.TrainingSetCollector
import java.awt.Toolkit

class Tester {

    private val modelTrainer = ModelTrainer()
    private val samplesCollector = ImageSamplesCollector(
        pxlStep = 50,
        slideSize = 60
    )

    @Test
    internal fun testRotate() {
        val rotated = rotate90(
            arrayOf(
                doubleArrayOf(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
            ), 3
        )
        assert(rotated[0].contentEquals(doubleArrayOf(6.0, 3.0, 0.0, 7.0, 4.0, 1.0, 8.0, 5.0, 2.0)))
    }

    @Test
    internal fun train() {
        TrainingSetCollector().train("/Users/oleg1024/Downloads/divan/", "/Users/oleg1024/Downloads/divan/heart")
    }


    @Test
    internal fun testLogitWithRespectToTheSizeOfTheInput() {
        val (X, y) = modelTrainer.getTrainingSet("/Users/oleg1024/Downloads/divan/heart", "logit_0.01")
        val f1 = mutableListOf<DoubleArray>()
        val accuracy = mutableListOf<DoubleArray>()
        for (i in 50..500 step 10) {
            val (subX, subY) = subSet(X, y, i)
            val classification = modelTrainer.test(subX, subY, "logit_0.01")
            f1.add(doubleArrayOf(i.toDouble(), classification.avg.f1))
            accuracy.add(doubleArrayOf(i.toDouble(), classification.avg.accuracy))
        }
        showPlot(f1.toTypedArray())
        showPlot(accuracy.toTypedArray())
        Thread.sleep(100000)
    }

    @Test
    internal fun testLogit() {
        modelTrainer.test("/Users/oleg1024/Downloads/divan/heart", "logit_0")
        modelTrainer.test("/Users/oleg1024/Downloads/divan/heart", "logit_0.01") //the best
        modelTrainer.test("/Users/oleg1024/Downloads/divan/heart", "logit_0.03")
    }

    @Test
    internal fun testWithDir() {

        val trainingSetFile = "/Users/oleg1024/Downloads/divan/heart"

//        val classifier = modelTrainer.train(trainingSetFile, "logit_0.01", 5000)
        val classifier = modelTrainer.train(trainingSetFile, "mlp_12", 5000)

        val centerAndScale = readFromFile<Pair<DoubleArray, DoubleArray>>("$trainingSetFile.options")
        val (center, scale) = centerAndScale

        val images = walkFileTree("/Users/oleg1024/Downloads/divan/")


        for (image in images) {
            val (scaleK, testImage) = resize(image, targetHeight = 400)
            val samples = samplesCollector.collect(testImage)
            val areaSums = samples
                .map { toDoubleArray(it.image).scale(center, scale) to it }
                .map { (array, sample) -> classifier.predict(array) to sample }
                .filter { (label, _) -> label == 1 }
                .map { (_, sample) -> sample.toArea() }
                .let { AreaSums(it) }

            val screenSize = Toolkit.getDefaultToolkit().screenSize.height.toDouble() * 0.8
            var (anotherScaleK, resultImage) = resize(image, screenSize.toInt())
            for (area in areaSums) {
                resultImage = highlightArea(resultImage, area.scaled(scaleK / anotherScaleK))
            }
            showImage(resultImage)
        }

        Thread.sleep(100000000)

    }


}
