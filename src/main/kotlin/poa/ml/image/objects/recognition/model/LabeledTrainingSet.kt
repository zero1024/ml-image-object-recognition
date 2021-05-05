package poa.ml.image.objects.recognition.model

import com.google.common.util.concurrent.AtomicDoubleArray
import poa.ml.image.objects.recognition.toDoubleArray
import smile.math.matrix.Matrix
import java.io.Serializable
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.sqrt


class LabeledTrainingSet(
    private val rows: CopyOnWriteArrayList<Pair<DoubleArray, Int>>
) : Serializable {

    constructor() : this(CopyOnWriteArrayList<Pair<DoubleArray, Int>>())

    @Volatile
    private var means: AtomicDoubleArray? = null

    @Volatile
    private var sumsq: AtomicDoubleArray? = null

    fun size() = rows.size

    fun positiveLabelsPercentage(): Int {
        val positive = rows.filter { it.second == 1 }.size
        return (positive * 100) / size()
    }

    fun add(samples: List<Pair<Sample, Boolean>>) {
        for ((sample, label) in samples) {
            val image = sample.image
            val array = toDoubleArray(image)
            addToMeans(array)
            addToSds(array)
            rows.add(array to if (label) 1 else 0)
        }
    }

    fun shuffle() = rows.shuffle()

    fun toMatrix(): Pair<Matrix, IntArray> {
        val X = rows.map { it.first }.toTypedArray()
        val y = rows.map { it.second }.toIntArray()
        return Matrix(X) to y
    }

    fun means(): DoubleArray {
        val res = DoubleArray(means!!.length())
        for (i in 0 until means!!.length()) {
            res[i] = means!![i] / size()
        }
        return res
    }

    fun sds(): DoubleArray {
        val res = DoubleArray(sumsq!!.length())
        for (i in 0 until sumsq!!.length()) {
            val mu = means!![i] / size()
            res[i] = sqrt(sumsq!![i] / size() - mu * mu)
        }
        return res
    }

    private fun addToMeans(array: DoubleArray) {
        if (means == null) {
            synchronized(this) {
                if (means == null) {
                    means = AtomicDoubleArray(array)
                }
            }
        } else {
            for ((j, v) in array.withIndex()) {
                means!!.addAndGet(j, v)
            }
        }
    }

    private fun addToSds(array: DoubleArray) {
        if (sumsq == null) {
            synchronized(this) {
                if (sumsq == null) {
                    sumsq = AtomicDoubleArray(array.map { it * it }.toDoubleArray())
                }
            }
        } else {
            for ((j, v) in array.withIndex()) {
                sumsq!!.addAndGet(j, v * v)
            }
        }
    }


}
