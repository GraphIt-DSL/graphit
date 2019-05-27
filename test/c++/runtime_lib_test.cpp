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

TEST_F(RuntimeLibTest, serialMSTTest) {
    WGraph wg = builtin_loadWeightedEdgesFromFile("../../test/graphs/mst_special_case.wel");
    NodeID start = 1;
    NodeID* parent_vector = minimum_spanning_tree(wg, start);

    for (int i = 0; i < wg.num_nodes(); i++){
        std::cout << "parent: " << parent_vector[i] << std::endl;
    }

    EXPECT_EQ (1 , parent_vector[1]);
    EXPECT_EQ (3 , parent_vector[4]);
    EXPECT_EQ (4 , parent_vector[5]);
}


TEST_F(RuntimeLibTest, serialMSTTest2) {
    WGraph wg = builtin_loadWeightedEdgesFromFile("../../test/graphs/test2.wel");
    NodeID start = 1;
    NodeID* parent_vector = minimum_spanning_tree(wg, start);

    for (int i = 0; i < wg.num_nodes(); i++){
        std::cout << "parent: " << parent_vector[i] << std::endl;
    }

    EXPECT_EQ (1 , parent_vector[2]);
    EXPECT_EQ (1 , parent_vector[3]);
    EXPECT_EQ (3 , parent_vector[4]);
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

TEST_F(RuntimeLibTest, IntersectSortedNodeSetHiroshiBasicTest){

    auto A = new NodeID[5]{1, 2, 3, 4, 5};
    auto B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectSortedNodeSetHiroshi(A, B, 5, 5);
    delete[] A;
    delete[] B;
    EXPECT_EQ(4, count);


}


TEST_F(RuntimeLibTest, IntersectSortedNodeSetBinarySearchBasicTest){

    auto A = new NodeID[5]{1, 2, 3, 4, 5};
    auto B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectSortedNodeSetBinarySearch(A, B, 5, 5);
    delete[] A;
    delete[] B;
    EXPECT_EQ(4, count);


}

TEST_F(RuntimeLibTest, IntersectSortedNodeSetMultipleSkipBasicTest){

    auto A = new NodeID[5]{1, 2, 3, 4, 5};
    auto B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectSortedNodeSetMultipleSkip(A, B, 5, 5);
    delete[] A;
    delete[] B;
    EXPECT_EQ(4, count);

}

TEST_F(RuntimeLibTest, IntersectSortedNodeSetNaiveBasicTest){

    auto A = new NodeID[5]{1, 2, 3, 4, 5};
    auto B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectSortedNodeSetNaive(A, B, 5, 5);
    delete[] A;
    delete[] B;
    EXPECT_EQ(4, count);

}


TEST_F(RuntimeLibTest, IntersectSortedNodeSetBitsetBasicTest){

    auto A = new NodeID[5]{1, 2, 3, 4, 5};
    auto B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectSortedNodeSetBitset(A, B, 5, 5);
    delete[] A;
    delete[] B;
    EXPECT_EQ(4, count);

}

TEST_F(RuntimeLibTest, IntersectSortedNodeSetCombinedBasicTest){

    auto A = new NodeID[5]{1, 2, 3, 4, 5};
    auto B = new NodeID[5]{1, 2, 4, 5, 6};
    size_t count = intersectSortedNodeSetCombined(A, B, 5, 5, 3, 0.2);
    delete[] A;
    delete[] B;
    EXPECT_EQ(4, count);

}

TEST_F(RuntimeLibTest, IntersectSortedNodeSetEmpty) {
    auto A = new NodeID[5]{1, 2, 3, 4, 5};
    auto B = new NodeID[5]{6, 7, 8, 9, 10};

    size_t countHiroshi = intersectSortedNodeSetHiroshi(A, B, 5, 5);
    size_t countBitset = intersectSortedNodeSetBitset(A, B, 5, 5);
    size_t countCombined1 = intersectSortedNodeSetCombined(A, B, 5, 5, 2, 0.1);
    size_t countCombined2 = intersectSortedNodeSetCombined(A, B, 5, 5, 200, 0.2);
    size_t countMultiSkip = intersectSortedNodeSetMultipleSkip(A, B, 5, 5);
    size_t countNaive = intersectSortedNodeSetNaive(A, B, 5, 5);

    delete[] A;
    delete[] B;

    EXPECT_EQ(0, countHiroshi);
    EXPECT_EQ(0, countBitset);
    EXPECT_EQ(0, countCombined1);
    EXPECT_EQ(0, countCombined2);
    EXPECT_EQ(0, countMultiSkip);
    EXPECT_EQ(0, countNaive);
}

TEST_F(RuntimeLibTest, IntersectSortedNodeSetLongerSets){
    auto A = new NodeID[234]{4, 8, 9, 11, 15, 22, 30, 31, 35, 37,
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

    auto B = new NodeID[240]{7, 14, 29, 34, 36, 68, 69, 78, 83, 107, 115,
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

    size_t countHiroshi = intersectSortedNodeSetHiroshi(A, B, 234, 240);
    size_t countBitset = intersectSortedNodeSetBitset(A, B, 234, 240);
    size_t countBinarySearch = intersectSortedNodeSetBinarySearch(B, A, 240, 234);
    size_t countBinarySearch2 = intersectSortedNodeSetBinarySearch(A, B, 234, 240);
    size_t countCombined1 = intersectSortedNodeSetCombined(A, B, 234, 240, 20, 0.1);
    size_t countCombined2 = intersectSortedNodeSetCombined(A, B, 234, 240, 5000, 0.2);
    size_t countMultiSkip = intersectSortedNodeSetMultipleSkip(A, B, 234, 240);
    size_t countNaive = intersectSortedNodeSetNaive(A, B, 234, 240);

    delete[] A;
    delete[] B;

    EXPECT_EQ(13, countHiroshi);
    EXPECT_EQ(13, countBitset);
    EXPECT_EQ(13, countCombined1);
    EXPECT_EQ(13, countCombined2);
    EXPECT_EQ(13, countBinarySearch);
    EXPECT_EQ(13, countBinarySearch2);
    EXPECT_EQ(13, countMultiSkip);
    EXPECT_EQ(13, countNaive);


}

TEST_F(RuntimeLibTest, IntersectSortedNodeSetOneSetEmpty){
    auto A = new NodeID[0]{};
    auto B = new NodeID[5]{3, 4, 23, 45, 56};

    ASSERT_DEATH(intersectSortedNodeSetHiroshi(A, B, 0, 5), ".*");

    size_t countBitset = intersectSortedNodeSetBitset(A, B, 0, 5);
    size_t countBinarySearch = intersectSortedNodeSetBinarySearch(A, B, 0, 5);
    size_t countCombined1 = intersectSortedNodeSetCombined(A, B, 0, 5, 1, 0.1);
    size_t countCombined2 = intersectSortedNodeSetCombined(A, B, 0, 5, 5000, 0.2);
    size_t countMultiSkip = intersectSortedNodeSetMultipleSkip(A, B, 0, 5);
    size_t countNaive = intersectSortedNodeSetNaive(A, B, 0, 5);

    delete[] A;
    delete[] B;


    EXPECT_EQ(0, countBitset);
    EXPECT_EQ(0, countBinarySearch);
    EXPECT_EQ(0, countCombined1);
    EXPECT_EQ(0, countCombined2);
    EXPECT_EQ(0, countMultiSkip);
    EXPECT_EQ(0, countNaive);

    auto A1 = new NodeID[5]{2, 3, 5, 6, 7};
    auto B1 = new NodeID[0]{};

    ASSERT_DEATH(intersectSortedNodeSetHiroshi(A1, B1, 5, 0), ".*");

    countBitset = intersectSortedNodeSetBitset(A1, B1, 5, 0);
    countBinarySearch = intersectSortedNodeSetBinarySearch(A1, B1, 5, 0);
    countCombined1 = intersectSortedNodeSetCombined(A1, B1, 5, 0, 1, 0.1);
    countCombined2 = intersectSortedNodeSetCombined(A1, B1, 5, 0, 5000, 0.2);
    countMultiSkip = intersectSortedNodeSetMultipleSkip(A1, B1, 5, 0);
    countNaive = intersectSortedNodeSetNaive(A1, B1, 5, 0);

    delete[] A1;
    delete[] B1;


    EXPECT_EQ(0, countBitset);
    EXPECT_EQ(0, countBinarySearch);
    EXPECT_EQ(0, countCombined1);
    EXPECT_EQ(0, countCombined2);
    EXPECT_EQ(0, countMultiSkip);
    EXPECT_EQ(0, countNaive);

}

TEST_F(RuntimeLibTest, IntersectSortedNodeSetLonger2) {

    auto A = new NodeID[379]{6, 14, 28, 59, 63, 69, 75, 77, 89, 94,
                             101, 105, 119, 134, 148, 158, 160, 167,
                             171, 188, 204, 208, 217, 222, 224, 226,
                             229, 235, 236, 250, 259, 272, 282, 285,
                             290, 296, 299, 307, 316, 317, 318, 325,
                             339, 341, 354, 377, 381, 382, 398, 408,
                             410, 413, 414, 416, 433, 438, 479, 481,
                             483, 492, 510, 511, 518, 520, 522, 527,
                             564, 584, 591, 612, 613, 624, 637, 644,
                             651, 662, 665, 672, 674, 690, 699, 704,
                             705, 713, 718, 719, 723, 738, 754, 755,
                             758, 765, 775, 782, 788, 813, 852, 872,
                             890, 894, 904, 907, 918, 925, 935, 947,
                             963, 1008, 1022, 1060, 1061, 1064, 1103,
                             1119, 1123, 1139, 1145, 1151, 1156, 1170,
                             1187, 1218, 1248, 1256, 1261, 1266, 1281,
                             1291, 1310, 1324, 1326, 1342, 1357, 1365,
                             1373, 1374, 1380, 1388, 1392, 1432, 1439,
                             1443, 1455, 1461, 1488, 1493, 1497, 1515,
                             1518, 1533, 1534, 1565, 1566, 1569, 1571,
                             1573, 1592, 1603, 1613, 1622, 1634, 1640,
                             1641, 1652, 1653, 1659, 1683, 1696, 1697,
                             1712, 1713, 1736, 1737, 1738, 1743, 1749,
                             1758, 1759, 1762, 1784, 1788, 1806, 1809,
                             1813, 1840, 1844, 1858, 1859, 1876, 1884,
                             1891, 1910, 1912, 1918, 1934, 1941, 1953,
                             1956, 1958, 1982, 1988, 2000, 2001, 2012,
                             2028, 2072, 2076, 2093, 2108, 2122, 2124,
                             2126, 2132, 2152, 2174, 2186, 2195, 2196,
                             2236, 2242, 2247, 2248, 2267, 2272, 2279,
                             2284, 2285, 2306, 2309, 2316, 2325, 2336,
                             2340, 2342, 2351, 2354, 2366, 2373, 2402,
                             2406, 2407, 2417, 2425, 2426, 2428, 2432,
                             2434, 2439, 2451, 2452, 2476, 2478, 2486,
                             2492, 2496, 2498, 2503, 2513, 2531, 2539,
                             2549, 2563, 2575, 2577, 2593, 2608, 2626,
                             2642, 2643, 2649, 2659, 2664, 2679, 2681,
                             2683, 2706, 2712, 2722, 2727, 2730, 2734,
                             2737, 2739, 2749, 2752, 2759, 2766, 2770,
                             2774, 2778, 2786, 2798, 2804, 2809, 2811,
                             2815, 2820, 2825, 2826, 2828, 2837, 2861,
                             2876, 2878, 2887, 2891, 2903, 2906, 2928,
                             2957, 2963, 2972, 2976, 2985, 2987, 2998,
                             3003, 3005, 3024, 3050, 3052, 3057, 3061,
                             3063, 3064, 3067, 3073, 3075, 3080, 3083,
                             3088, 3094, 3102, 3104, 3112, 3122, 3131,
                             3144, 3151, 3161, 3169, 3175, 3181, 3183,
                             3186, 3192, 3197, 3205, 3208, 3211, 3213,
                             3219, 3234, 3238, 3244, 3257, 3258, 3266,
                             3290, 3311, 3314, 3329, 3346, 3373, 3380,
                             3387, 3403, 3414, 3416, 3427, 3428, 3429,
                             3431, 3433, 3444, 3449, 3452, 3456, 3464};

    auto B = new NodeID[612]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                             14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26,
                             27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38,
                             39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                             50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                             61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73,
                             74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85,
                             86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
                             97, 98, 100, 101, 102, 103, 104, 105, 106,
                             107, 108, 109, 110, 111, 112, 113, 114, 115,
                             116, 117, 118, 119, 120, 121, 122, 123, 126,
                             128, 129, 130, 131, 132, 133, 134, 136, 137,
                             138, 139, 140, 141, 142, 143, 144, 145, 146,
                             147, 148, 149, 150, 151, 152, 153, 154, 155,
                             156, 157, 158, 159, 160, 161, 162, 163, 164,
                             165, 166, 167, 169, 170, 171, 172, 173, 174,
                             175, 176, 177, 178, 179, 180, 181, 182, 183,
                             184, 185, 187, 188, 190, 192, 193, 194, 195,
                             197, 198, 199, 200, 201, 204, 205, 206, 207,
                             208, 209, 210, 211, 212, 213, 214, 215, 216,
                             217, 218, 219, 221, 222, 223, 224, 225, 226,
                             227, 228, 229, 230, 231, 232, 234, 235, 236,
                             237, 238, 239, 240, 241, 242, 243, 244, 245,
                             246, 248, 249, 251, 252, 253, 255, 256, 257,
                             258, 259, 260, 261, 262, 263, 264, 265, 267,
                             268, 272, 273, 274, 275, 276, 277, 278, 279,
                             280, 281, 282, 283, 284, 285, 286, 287, 288,
                             289, 290, 291, 292, 293, 294, 295, 296, 297,
                             298, 299, 301, 302, 303, 304, 305, 306, 307,
                             308, 309, 310, 311, 312, 313, 314, 315, 316,
                             318, 319, 320, 321, 322, 323, 325, 326, 327,
                             328, 329, 330, 331, 332, 333, 334, 337, 338,
                             340, 341, 342, 345, 346, 347, 348, 349, 350,
                             351, 353, 354, 355, 357, 358, 359, 361, 362,
                             363, 365, 366, 367, 368, 370, 371, 372, 373,
                             374, 376, 378, 379, 380, 381, 382, 383, 384,
                             385, 386, 387, 388, 389, 390, 391, 393, 394,
                             396, 397, 398, 399, 400, 401, 402, 403, 404,
                             405, 407, 408, 409, 410, 411, 412, 413, 414,
                             415, 416, 419, 420, 422, 423, 424, 425, 426,
                             427, 428, 429, 431, 432, 433, 435, 437, 438,
                             440, 441, 443, 444, 445, 446, 447, 448, 449,
                             450, 451, 452, 453, 454, 455, 456, 459, 460,
                             461, 462, 463, 464, 465, 466, 467, 469, 470,
                             471, 472, 474, 475, 476, 477, 478, 479, 480,
                             481, 482, 483, 484, 485, 486, 487, 488, 489,
                             490, 491, 492, 493, 495, 496, 497, 498, 499,
                             500, 501, 502, 503, 504, 505, 506, 507, 508,
                             509, 510, 511, 512, 513, 514, 515, 516, 517,
                             519, 520, 521, 523, 524, 526, 527, 528, 529,
                             530, 531, 532, 533, 534, 535, 536, 537, 538,
                             539, 540, 541, 542, 544, 545, 546, 547, 548,
                             550, 551, 553, 554, 555, 556, 557, 558, 559,
                             560, 561, 562, 564, 565, 566, 567, 568, 569,
                             570, 571, 572, 573, 574, 575, 576, 577, 578,
                             579, 580, 581, 582, 583, 584, 585, 586, 587,
                             588, 589, 591, 592, 593, 594, 595, 596, 598,
                             599, 600, 601, 602, 604, 605, 607, 608, 610,
                             611, 612, 613, 616, 617, 618, 620, 621, 622,
                             624, 625, 626, 627, 628, 630, 632, 634, 635,
                             636, 637, 638, 639, 640, 642, 643, 644, 645,
                             646, 647, 648, 649, 650, 651, 652, 653, 654,
                             655, 656, 657, 658, 659, 660, 661, 662, 663,
                             664, 665, 668, 671, 672, 673, 674, 675, 676,
                             677, 678, 679, 680, 681, 683, 684, 685, 686,
                             687, 688, 689, 690, 691, 692, 693, 694, 696};


    size_t countBitset = intersectSortedNodeSetBitset(A, B, 379, 612);
    size_t countBinarySearch = intersectSortedNodeSetBinarySearch(A, B, 379, 612);
    size_t countCombined1 = intersectSortedNodeSetCombined(A, B, 379, 612, 100, 0.3);
    size_t countCombined2 = intersectSortedNodeSetCombined(A, B, 379, 612, 5000, 0.9);
    size_t countMultiSkip = intersectSortedNodeSetMultipleSkip(A, B, 379, 612);
    size_t countNaive = intersectSortedNodeSetNaive(A, B, 379, 612);


    delete[] A;
    delete[] B;

    EXPECT_EQ(73, countBitset);
    EXPECT_EQ(73, countBinarySearch);
    EXPECT_EQ(73, countCombined1);
    EXPECT_EQ(73, countCombined2);
    EXPECT_EQ(73, countMultiSkip);
    EXPECT_EQ(73, countNaive);
}

TEST_F(RuntimeLibTest, GetRandomOutNeighborTest) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    NodeID ngh = g.get_random_out_neigh(1);
    std::cout << ngh << std::endl;
    EXPECT_LE (ngh , 5);
    EXPECT_GE (ngh , 2);
}

TEST_F(RuntimeLibTest, GetRandomInNeighborTest) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    NodeID ngh = g.get_random_in_neigh(4);
    std::cout << ngh << std::endl;
    EXPECT_LE (ngh , 3);
    EXPECT_GE (ngh , 1);

}

