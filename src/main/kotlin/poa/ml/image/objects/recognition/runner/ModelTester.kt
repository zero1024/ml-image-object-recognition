package poa.ml.image.objects.recognition.runner

import poa.ml.image.objects.recognition.readFromFile
import smile.base.mlp.Layer
import smile.base.mlp.OutputFunction
import smile.classification.mlp
import smile.math.matrix.Matrix

class ModelTester {

    fun test(trainingSetFile: String) {
        val X = readFromFile<Matrix>("$trainingSetFile.X")
        val y = readFromFile<IntArray>("$trainingSetFile.y")

        val center = X.colMeans()
        val scale = X.colSds()

        val xScaled = X.scale(center, scale).toArray()

        val classifier =
            mlp(xScaled, y, arrayOf(Layer.sigmoid(18), Layer.sigmoid(9), Layer.mle(1, OutputFunction.SIGMOID)))
        println(classifier)
    }

}