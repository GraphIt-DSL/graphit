//
// Created by Yunming Zhang on 4/27/17.
//

#include <gtest.h>
#include "intrinsics.h"
#include "infra_gapbs/graph_verifier.h"
#include "infra_gapbs/intersections.h"




class RuntimeLibTest : public ::testing::Test {
protected:
    virtual void SetUp() {

    }

    virtual void TearDown() {

    }

};


TEST_F(RuntimeLibTest, SimpleLoadGraphFromFileTest) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    EXPECT_EQ (7 , g.num_edges());
}

TEST_F(RuntimeLibTest, SimpleLoadVerticesromEdges) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    int num_vertices = builtin_getVertices(g);
    //max node id + 1, assumes the first node has id 0
    EXPECT_EQ (5 , num_vertices);
}

TEST_F(RuntimeLibTest, GetOutDegrees) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    auto out_degrees = builtin_getOutDegrees(g);
    //max node id + 1, assumes the first node has id 0
    //TODO: fix this, we don't use vectors anymore
    //EXPECT_EQ (5 , out_degrees.size());
}

TEST_F(RuntimeLibTest, TimerTest) {
//    float start_time = getTime();
//    sleep(1);
//    float end_time = getTime();
    startTimer();
    sleep(1);
    float elapsed_time = stopTimer();
    std::cout << "elapsed_time: " << elapsed_time << std::endl;

    startTimer();
    sleep(2);
    elapsed_time = stopTimer();
    std::cout << "elapsed_time: " << elapsed_time << std::endl;
    EXPECT_EQ (5 , 5);
}

TEST_F(RuntimeLibTest, VertexSubsetSimpleTest) {
    bool test_flag = true;
    auto vertexSubset = new VertexSubset<int>(5, 0);
    for (int v = 0; v < 5; v = v+2){
        vertexSubset->addVertex(v);
    }

    for (int v = 1; v < 5; v = v+2){
        if (vertexSubset->contains(v))
            test_flag = false;
    }

    for (int v = 0; v < 5; v = v+2){
        if (!vertexSubset->contains(v))
            test_flag = false;
    }

    EXPECT_EQ(builtin_getVertexSetSize(vertexSubset), 3);


    delete vertexSubset;

    EXPECT_EQ(true, test_flag);

} 

TEST_F(RuntimeLibTest, IntersectHiroshiBasicTest){

    NodeID* A = new NodeID[5]{1, 2, 3, 4, 5};
    NodeID* B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectHiroshi(A, B, 5, 5);
    delete A;
    delete B;
    EXPECT_EQ(4, count);


}


TEST_F(RuntimeLibTest, IntersectBinarySearchBasicTest){

    NodeID* A = new NodeID[5]{1, 2, 3, 4, 5};
    NodeID* B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectBinarySearch(A, B, 5, 5);
    delete A;
    delete B;
    EXPECT_EQ(4, count);


}

TEST_F(RuntimeLibTest, IntersectMultipleSkipBasicTest){

    NodeID* A = new NodeID[5]{1, 2, 3, 4, 5};
    NodeID* B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectMultipleSkip(A, B, 5, 5);
    delete A;
    delete B;
    EXPECT_EQ(4, count);

}

TEST_F(RuntimeLibTest, IntersectNaiveBasicTest){

    NodeID* A = new NodeID[5]{1, 2, 3, 4, 5};
    NodeID* B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectNaive(A, B, 5, 5);
    delete A;
    delete B;
    EXPECT_EQ(4, count);

}


TEST_F(RuntimeLibTest, IntersectBitsetBasicTest){

    NodeID* A = new NodeID[5]{1, 2, 3, 4, 5};
    NodeID* B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectBitset(A, B, 5, 5);
    delete A;
    delete B;
    EXPECT_EQ(4, count);

}

TEST_F(RuntimeLibTest, IntersectCombinedBasicTest){

    NodeID* A = new NodeID[5]{1, 2, 3, 4, 5};
    NodeID* B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectCombined(A, B, 5, 5, 0.2, 3);
    delete A;
    delete B;
    EXPECT_EQ(4, count);

}

TEST_F(RuntimeLibTest, IntersectEmpty) {
    NodeID* A = new NodeID[5]{1, 2, 3, 4, 5};
    NodeID* B = new NodeID[5]{6, 7, 8, 9, 10};

    size_t countHiroshi = intersectHiroshi(A, B, 5, 5);
    size_t countBitset = intersectBitset(A, B, 5, 5);
    size_t countCombined1 = intersectCombined(A, B, 5, 5, 0.1, 2);
    size_t countCombined2 = intersectCombined(A, B, 5, 5, 0.2, 200);
    size_t countMultiSkip = intersectMultipleSkip(A, B, 5, 5);
    size_t countNaive = intersectNaive(A, B, 5, 5);

    delete A;
    delete B;

    EXPECT_EQ(0, countHiroshi);
    EXPECT_EQ(0, countBitset);
    EXPECT_EQ(0, countCombined1);
    EXPECT_EQ(0, countCombined2);
    EXPECT_EQ(0, countMultiSkip);
    EXPECT_EQ(0, countNaive);
}

TEST_F(RuntimeLibTest, IntersectLongerSets){
    NodeID* A = new NodeID[234]{4, 8, 9, 11, 15, 22, 30, 31, 35, 37, 
                            39, 44, 47, 66, 67, 69, 95, 112, 125, 
                            127, 131, 136, 139, 152, 155, 200, 234, 
                            237, 251, 269, 282, 286, 301, 302, 311, 
                            313, 325, 342, 345, 346, 349, 352, 355, 
                            369, 408, 418, 427, 431, 437, 438, 448, 
                            450, 462, 466, 494, 502, 505, 509, 514, 
                            532, 541, 547, 561, 567, 573, 582, 591, 
                            600, 610, 625, 626, 636, 660, 664, 680, 
                            684, 705, 715, 718, 743, 744, 757, 770, 
                            776, 779, 781, 783, 799, 817, 818, 829, 
                            840, 844, 845, 877, 880, 881, 882, 887, 
                            915, 917, 920, 933, 943, 945, 961, 964, 
                            965, 998, 1002, 1007, 1009, 1020, 1023, 
                            1030, 1033, 1043, 1046, 1049, 1086, 1091, 
                            1106, 1115, 1129, 1132, 1136, 1141, 1149,
                            1161, 1164, 1172, 1174, 1175, 1179, 1183, 
                            1185, 1186, 1193, 1201, 1211, 1230, 1245,
                            1271, 1272, 1277, 1278, 1293, 1326, 1346, 
                            1350, 1358, 1360, 1374, 1386, 1392, 1396, 
                            1398, 1412, 1414, 1416, 1429, 1434, 1446, 
                            1453, 1455, 1463, 1475, 1478, 1479, 1487, 
                            1488, 1502, 1503, 1509, 1518, 1541, 1543, 
                            1548, 1557, 1574, 1577, 1583, 1589, 1592, 
                            1606, 1607, 1608, 1620, 1649, 1654, 1657, 
                            1672, 1693, 1697, 1704, 1711, 1713, 1719, 
                            1729, 1733, 1739, 1752, 1753, 1762, 1769, 
                            1775, 1792, 1793, 1799, 1800, 1818, 1823, 
                            1827, 1828, 1832, 1837, 1846, 1847, 1849, 
                            1859, 1860, 1861, 1864, 1886, 1891, 1918, 
                            1926, 1931, 1934, 1951, 1958, 1966, 1982, 1999};

    NodeID* B = new NodeID[240]{7, 14, 29, 34, 36, 68, 69, 78, 83, 107, 115, 
                               120, 130, 132, 146, 176, 180, 183, 192, 203, 
                               212, 215, 268, 274, 280, 285, 295, 302, 323, 
                               344, 352, 356, 387, 411, 435, 437, 450, 484, 
                               487, 499, 500, 504, 505, 519, 522, 528, 544, 
                               548, 549, 567, 568, 574, 581, 587, 595, 605, 
                               613, 616, 624, 631, 635, 639, 643, 645, 653, 
                               658, 660, 668, 670, 674, 692, 695, 700, 724, 
                               737, 771, 772, 797, 826, 833, 864, 873, 886, 
                               891, 909, 916, 927, 947, 956, 976, 977, 1031, 
                               1040, 1089, 1098, 1115, 1126, 1133, 1163, 1171, 
                               1185, 1219, 1223, 1238, 1243, 1292, 1310, 1325, 
                               1332, 1351, 1352, 1353, 1364, 1369, 1376, 1390, 
                               1401, 1417, 1425, 1459, 1464, 1472, 1480, 1556, 
                               1571, 1588, 1624, 1626, 1643, 1667, 1685, 1689, 
                               1694, 1698, 1699, 1710, 1752, 1772, 1797, 1811, 
                               1813, 1831, 1833, 1834, 1846, 1848, 1856, 1861, 
                               1879, 1880, 1885, 1897, 1903, 1905, 1928, 1942, 
                               1944, 1948, 1961, 1981, 1984, 1985, 1987, 2007, 
                               2020, 2042, 2050, 2051, 2055, 2060, 2118, 2130, 
                               2157, 2185, 2187, 2191, 2235, 2248, 2249, 2250, 
                               2254, 2256, 2258, 2267, 2285, 2287, 2320, 2321, 
                               2327, 2345, 2369, 2376, 2386, 2400, 2418, 2424, 
                               2444, 2458, 2472, 2474, 2475, 2503, 2507, 2527, 
                               2528, 2535, 2539, 2559, 2599, 2600, 2601, 2609, 
                               2611, 2621, 2622, 2623, 2637, 2686, 2687, 2714, 
                               2754, 2757, 2768, 2771, 2778, 2791, 2811, 2827, 
                               2834, 2839, 2852, 2866, 2870, 2888, 2897, 2906, 
                               2908, 2932, 2938, 2972};

    size_t countHiroshi = intersectHiroshi(A, B, 234, 240);
    size_t countBitset = intersectBitset(A, B, 234, 240);
    size_t countBinarySearch = intersectBinarySearch(B, A, 240, 234);
    size_t countBinarySearch2 = intersectBinarySearch(A, B, 234, 240);
    size_t countCombined1 = intersectCombined(A, B, 234, 240, 0.1, 20);
    size_t countCombined2 = intersectCombined(A, B, 234, 240, 0.2, 5000);
    size_t countMultiSkip = intersectMultipleSkip(A, B, 234, 240);
    size_t countNaive = intersectNaive(A, B, 234, 240);

    delete A;
    delete B;

    EXPECT_EQ(13, countHiroshi);
    EXPECT_EQ(13, countBitset);
    EXPECT_EQ(13, countCombined1);
    EXPECT_EQ(13, countCombined2);
    EXPECT_EQ(13, countBinarySearch);
    EXPECT_EQ(13, countBinarySearch2);    
    EXPECT_EQ(13, countMultiSkip);
    EXPECT_EQ(13, countNaive);

    
}

TEST_F(RuntimeLibTest, IntersectOneSetEmpty){
    NodeID* A = new NodeID[0]{};
    NodeID* B = new NodeID[5]{3, 4, 23, 45, 56};

    ASSERT_DEATH(intersectHiroshi(A, B, 0, 5), ".*");

    size_t countBitset = intersectBitset(A, B, 0, 5);
    size_t countBinarySearch = intersectBinarySearch(A, B, 0, 5);
    size_t countCombined1 = intersectCombined(A, B, 0, 5, 0.1, 1);
    size_t countCombined2 = intersectCombined(A, B, 0, 5, 0.2, 5000);
    size_t countMultiSkip = intersectMultipleSkip(A, B, 0, 5);
    size_t countNaive = intersectNaive(A, B, 0, 5);
    
    delete A;
    delete B;

    
    EXPECT_EQ(0, countBitset);
    EXPECT_EQ(0, countBinarySearch);
    EXPECT_EQ(0, countCombined1);
    EXPECT_EQ(0, countCombined2);
    EXPECT_EQ(0, countMultiSkip);
    EXPECT_EQ(0, countNaive);

    NodeID* A1 = new NodeID[5]{2, 3, 5, 6, 7};
    NodeID* B1 = new NodeID[0]{};

    ASSERT_DEATH(intersectHiroshi(A1, B1, 5, 0), ".*");

    countBitset = intersectBitset(A1, B1, 5, 0);
    countBinarySearch = intersectBinarySearch(A1, B1, 5, 0);
    countCombined1 = intersectCombined(A1, B1, 5, 0, 0.1, 1);
    countCombined2 = intersectCombined(A1, B1, 5, 0, 0.2, 5000);
    countMultiSkip = intersectMultipleSkip(A1, B1, 5, 0);
    countNaive = intersectNaive(A1, B1, 5, 0);

    delete A1;
    delete B1;

    
    EXPECT_EQ(0, countBitset);
    EXPECT_EQ(0, countBinarySearch);
    EXPECT_EQ(0, countCombined1);
    EXPECT_EQ(0, countCombined2);
    EXPECT_EQ(0, countMultiSkip);
    EXPECT_EQ(0, countNaive);

}