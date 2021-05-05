package poa.ml.image.objects.recognition.runner

import poa.ml.image.objects.recognition.printlnEnd
import poa.ml.image.objects.recognition.printlnStart
import poa.ml.image.objects.recognition.readFromFile
import poa.ml.image.objects.recognition.subSet
import smile.base.mlp.Layer
import smile.base.mlp.LayerBuilder
import smile.base.mlp.OutputFunction
import smile.classification.Classifier
import smile.classification.logit
import smile.classification.mlp
import smile.math.matrix.Matrix
import smile.validation.ClassificationValidations
import smile.validation.CrossValidation
import java.util.function.BiFunction

class ModelTrainer {

    fun test(
        trainingSetFile: String,
        classifierCode: String,
        nrows: Int = -1
    ): ClassificationValidations<Classifier<DoubleArray>> {
        val (xArray, yArray) = getTrainingSet(trainingSetFile, nrows)
        return test(xArray, yArray, classifierCode)
    }

    fun test(
        xArray: Matrix,
        yArray: IntArray,
        classifierCode: String
    ): ClassificationValidations<Classifier<DoubleArray>> {
        printlnStart("===Training $classifierCode classifier with ${yArray.size} examples...")
        val classification = CrossValidation.classification(5, xArray.toArray(), yArray, dispatch(classifierCode))
        println("")
        println("")
        println("")
        println(classification.avg)
        println("")
        println("")
        println("")
        printlnEnd("===Done training.")
        return classification
    }

    fun train(
        trainingSetFile: String,
        classifierCode: String,
        nrows: Int = -1
    ): Classifier<DoubleArray> {
        val (xArray, yArray) = getTrainingSet(trainingSetFile, nrows)
        printlnStart("===Training $classifierCode classifier with ${yArray.size} examples...")
        val res = dispatch(classifierCode).apply(xArray.toArray(), yArray)
        printlnEnd("===Done training.")
        return res
    }

    fun getTrainingSet(trainingSetFile: String, nrows: Int = -1): Pair<Matrix, IntArray> {
        printlnStart("===Reading the matrix...")
        val X = readFromFile<Matrix>("$trainingSetFile.X")
        printlnEnd("===Done reading. Size = ${X.nrows()}x${X.ncols()}")
        printlnStart("===Reading the labels...")
        val y = readFromFile<IntArray>("$trainingSetFile.y")
        printlnEnd("===Done reading.")
        return subSet(X, y, nrows)
    }


}

private fun dispatch(name: String): BiFunction<Array<DoubleArray>, IntArray, Classifier<DoubleArray>> {
    return when {
        name.startsWith("mlp.") -> {
            val layers = name.split(".")
                .mapNotNull { it.toIntOrNull() }
                .map { Layer.sigmoid(it) as LayerBuilder }
                .toMutableList()
            layers.add(Layer.mle(1, OutputFunction.SIGMOID))
            BiFunction { X, y -> mlp(X, y, layers.toTypedArray(), weightDecay = 0.3) }
        }
        name.startsWith("logit.") -> {
            val lambda = name.split("logit.")[1]
            BiFunction { X, y -> logit(X, y, lambda.toDouble()) }
        }
        else -> {
            throw IllegalStateException()
        }
    }
}

