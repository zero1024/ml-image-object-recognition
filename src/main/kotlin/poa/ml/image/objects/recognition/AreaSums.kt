package poa.ml.image.objects.recognition

import java.awt.geom.Area

class AreaSums(private val areaList: List<Area>) : Iterable<Area> {

    override fun iterator(): Iterator<Area> {
        return getSums().iterator()
    }

    private fun getSums() = getSums(areaList)

    private fun getSums(sources: List<Area>): List<Area> {
        val results = mutableListOf<Area>()
        for (source in sources) {

            val anyNeighbor = results.firstOrNull { it.isIntersect(source) }

            if (anyNeighbor !== null) {
                anyNeighbor.add(source.copy())
            } else {
                results.add(source.copy())
            }
        }
        return if (results.size == sources.size) {
            results
        } else {
            getSums(results)
        }
    }

    private fun Area.isIntersect(area: Area): Boolean {
        val res = copy()
        res.add(area.copy())
        return res.isSingular
    }


}