package poa.ml.image.objects.recognition.model

import org.apache.commons.lang3.SerializationUtils.deserialize
import org.apache.commons.lang3.SerializationUtils.serialize
import poa.ml.image.objects.recognition.toDoubleArray
import smile.math.matrix.Matrix
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.concurrent.CopyOnWriteArrayList

class LabeledTrainingSet(
    private val rows: CopyOnWriteArrayList<DoubleArray>,
    private val labels: CopyOnWriteArrayList<Int>
) {

    constructor() : this(CopyOnWriteArrayList<DoubleArray>(), CopyOnWriteArrayList<Int>())

    fun size() = rows.size

    fun positiveLabelsPercentage(): Int {
        val positive = labels.filter { it == 1 }.size
        return (positive * 100) / size()
    }

    fun add(samples: List<Pair<Sample, Boolean>>) {
        for ((sample, label) in samples) {
            labels.add(if (label) 1 else 0)
            val image = sample.image
            val array = toDoubleArray(image)
            rows.add(array)
        }
    }

    fun toMatrix(): Pair<Matrix, IntArray> {
        return Matrix(rows.toTypedArray()) to labels.toIntArray()
    }

    fun save(file: String) {
        FileOutputStream(file).use {
            serialize(rows to labels, it)
        }
    }

    companion object {
        fun fromFile(file: String): LabeledTrainingSet {
            FileInputStream(file).use {
                val (rows, labels) = deserialize(it) as Pair<CopyOnWriteArrayList<DoubleArray>, CopyOnWriteArrayList<Int>>
                return LabeledTrainingSet(rows, labels)

            }
        }
    }

}