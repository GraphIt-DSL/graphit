// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Converts a Ligra hypergraph in adjacency graph format into edge list format

#include "parseCommandLine.h"
#include "graphIO.h"
#include "parallel.h"
#include <iostream>
#include <sstream>
using namespace benchIO;
using namespace std;

int parallel_main(int argc, char* argv[]) {
  commandLine P(argc,argv," [-w] <inFile>");
  char* iFile = P.getArgument(0);
  bool weighted = P.getOptionValue("-w");
  char* outS;
  if (weighted){
    outS = (char*) ".wel";
  } else {
    outS = (char*) ".el";
  }
  char outFile[strlen(iFile) + strlen(outS) + 1];
  *outFile = '\0';
  strcat(outFile, iFile);
  strcat(outFile, outS); 

  if(!weighted) {
    hypergraph<uintT> G = readHypergraphFromFile<uintT>(iFile);
    // config << G.nv << " " << G.mv << " " << G.nh << " " << G.mh;
    pair<uintT,uintT>* edgelist = (pair<uintT,uintT>*) malloc(sizeof(pair<uintT,uintT>) * 2 * G.mv);
    uintE en = 0; 
    for (uintT u = 0; u < G.nv; u++){
        intT deg = G.V[u].degree;
      for (intT i = 0; i < deg; i++) {
            edgelist[en] = pair<uintT,uintT>((uintT) u, (uintT) G.V[u].Neighbors[i] + G.nv);
            en++;
        }
    }

    for (uintT h = 0; h < G.nh; h++){
        intT deg = G.H[h].degree;
        for (intT i = 0; i < deg; i++) {
            edgelist[en] = pair<uintT,uintT>((uintT) h + G.nv, (uintT) G.H[h].Neighbors[i]);
            en++;
        }
    }
    std::cout << en << " edges written to edgelist!" << std::endl;
    writeArrayToFile("", edgelist, 2 * G.mv, outFile); 
  }
  else {
    wghHypergraph<uintT> G = readWghHypergraphFromFile<uintT>(iFile);
    pair<uintT,pair<uintT, uintT>>* edgelist = (pair<uintT,pair<uintT,uintT>>*) malloc(sizeof(pair<uintT,pair<uintT, uintT>>) * 2 * G.mv);
    uintE en = 0;

    for (uintT u = 0; u < G.nv; u++){
        intT deg = G.V[u].degree;
      for (intT i = 0; i < deg; i++) {
            edgelist[en] = pair<uintT,pair<uintT, uintT>>((uintT) u, pair<uintT, uintT>((uintT) G.V[u].Neighbors[i] + G.nv, (uintT) G.V[u].nghWeights[i]));
            en++;
        }
    }

    for (uintT h = 0; h < G.nh; h++){
        intT deg = G.H[h].degree;
        for (intT i = 0; i < deg; i++) {
            edgelist[en] = pair<uintT,pair<uintT,uintT>>((uintT) h + G.nv, pair<uintT, uintT>((uintT) G.H[h].Neighbors[i], (uintT) G.H[h].nghWeights[i]));
            en++;
        }
    }
    std::cout << en << " edges written to edgelist!" << std::endl;
    writeArrayToFile("", edgelist, 2 * G.mv, outFile); 
  }

}