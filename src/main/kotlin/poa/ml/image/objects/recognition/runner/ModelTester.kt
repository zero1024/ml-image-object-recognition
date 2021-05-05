package poa.ml.image.objects.recognition.runner

import poa.ml.image.objects.recognition.readFromFile
import smile.base.mlp.Layer
import smile.base.mlp.OutputFunction
import smile.classification.mlp
import smile.math.matrix.Matrix

class ModelTester {

    fun test(trainingSetFile: String) {
        println("===Reading the matrix...")
        val X = readFromFile<Matrix>("$trainingSetFile.X")
        println("===Done reading. Size = ${X.size()}")
        println("")
        println("===Reading the labels...")
        val y = readFromFile<IntArray>("$trainingSetFile.y")
        println("===Done reading.")
        println("")

        val (center, scale) = readFromFile<Pair<DoubleArray, DoubleArray>>("$trainingSetFile.options")

        val x = X.toArray()
        println("===Training...")
        val classifier =
            mlp(x, y, arrayOf(Layer.sigmoid(18), Layer.sigmoid(9), Layer.mle(1, OutputFunction.SIGMOID)))
        println(classifier)
        println("===Done training.")
    }

}