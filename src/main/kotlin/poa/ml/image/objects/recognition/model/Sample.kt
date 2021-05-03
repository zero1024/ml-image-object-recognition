package poa.ml.image.objects.recognition.model

import java.awt.Rectangle
import java.awt.geom.Area
import java.awt.image.BufferedImage

data class Sample(val x: Int, val y: Int, val image: BufferedImage) {

    private val area = Area(Rectangle(x, y, image.width, image.height))

    fun toArea() = area.clone() as Area

}
