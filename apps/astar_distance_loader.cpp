#include <iostream>
#include <limits>
#include <queue>
#include <vector>

#include "intrinsics.h"

struct Coords {
  double lat;
  double lon;
};



const double EARTH_RADIUS_CM = 637100000.0;
pvector<Coords> *coords;

WeightT dist(pvector<Coords> const &coords,
             NodeID source, NodeID target) {
  // Use the haversine formula to compute the great-angle radians
  double latS = std::sin(coords[source].lat - coords[target].lat);
  double lonS = std::sin(coords[source].lon - coords[target].lon);
  double a = latS * latS +
             lonS * lonS * std::cos(coords[source].lat) * std::cos(coords[target].lat);
  double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));

  return c * EARTH_RADIUS_CM;
}

void readLatsLons(std::string const &filename, pvector<Coords> &coords) {
  std::ifstream in;
  in.open(filename, std::ios::binary);

  // Entry reading utilities
  auto readU = [&]() -> uint32_t {
    union U {
      uint32_t val;
      char bytes[sizeof(uint32_t)];
    };
    U u;
    in.read(u.bytes, sizeof(uint32_t));
    assert(!in.fail());
    return u.val;
  };

  auto readD = [&]() -> double {
    union U {
      double val;
      char bytes[sizeof(double)];
    };
    U u;
    in.read(u.bytes, sizeof(double));
    assert(!in.fail());
    return u.val;
  };

  uint32_t magic = readU();
  assert(magic == 0x150842A7);

  uint32_t numNodes = readU();
  for (uint32_t i = 0; i < numNodes; i++) {
    coords[i].lat = readD();
    coords[i].lon = readD();

    uint32_t numNeighbors = readU();
    for (uint32_t j = 0; j < numNeighbors; j++)
      readU();
    for (uint32_t j = 0; j < numNeighbors; j++)
      readD();
  }
}


extern void load_coords (std::string filename, int num_nodes); 
extern double calculate_distance (NodeID source, NodeID destination); 

void load_coords(std::string filename, int num_nodes) {
	coords = new pvector<Coords>(num_nodes);
	readLatsLons(filename, *coords);
	return;
}


double calculate_distance(NodeID source, NodeID destination) {
	return dist(*coords, source, destination);
}
