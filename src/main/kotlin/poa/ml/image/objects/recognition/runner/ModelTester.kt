package poa.ml.image.objects.recognition.runner

import poa.ml.image.objects.recognition.model.LabeledTrainingSet

class ModelTester {

    fun test(trainingSetFile: String) {
        val trainingSet = LabeledTrainingSet.fromFile(trainingSetFile)
        println(trainingSet)

    }

}