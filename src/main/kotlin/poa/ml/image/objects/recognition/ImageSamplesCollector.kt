package poa.ml.image.objects.recognition

import poa.ml.image.objects.recognition.model.Sample
import java.awt.Rectangle
import java.awt.geom.Area
import java.awt.image.BufferedImage
import kotlin.math.min

class ImageSamplesCollector(
    private val pxlStep: Int = 5,
    private val slideSize: Int = 50
) {


    private val ignorablePxlStep = slideSize / 5

    init {
        assert(slideSize >= pxlStep) { "slide size should be greater than or equal to pxl step" }
    }

    fun collect(image: BufferedImage, filter: (Int, Int) -> Boolean = { _, _ -> true }): List<Sample> {
        if (image.width < slideSize || image.height < slideSize)
            return emptyList()
        val list = mutableListOf<Sample>()
        for (y in 0 until image.height step pxlStep) {
            for (x in 0 until image.width step pxlStep) {
                val resultX = min(x, image.width - slideSize)
                val resultY = min(y, image.height - slideSize)
                if (x < image.width - ignorablePxlStep &&
                    y < image.height - ignorablePxlStep &&
                    filter(resultX, resultY)
                ) {
                    val sub = image.getSubimage(
                        resultX,
                        resultY,
                        slideSize,
                        slideSize
                    )
                    list.add(Sample(resultX, resultY, sub))
                }
            }
        }
        return list
    }

    fun collect(image: BufferedImage, area: Area): List<Sample> {
        val rectangle = area.bounds
        val imagePart = image.getSubimage(rectangle.x, rectangle.y, rectangle.width, rectangle.height)
        return collect(imagePart) { x, y -> area.contains(Rectangle(x, y, slideSize, slideSize)) }
    }

}

