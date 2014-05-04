/*
 * main.cpp
 *
 *  Created on: Apr 9, 2014
 *      Author: cve
 */

//
//
//#include <iostream>
//#include <cuv.hpp>
//#include <vector>
//
//using namespace cuv;
//using namespace std;


//#define BOOST_TEST_MODULE CuvTest
//#include <boost/test/included/unit_test.hpp>
//
//BOOST_AUTO_TEST_CASE(test_tensor_dim)
//{
//	{
//		tensor<float, host_memory_space> t(extents[5]);
//		BOOST_CHECK_EQUAL(1, t.ndim());
//	}
//
//	{
//
//		tensor<float, host_memory_space> matrix(extents[3][5][10]);
//		tensor<double, host_memory_space> a(extents[10]);
//		tensor<double, host_memory_space> b(extents[10]);
//		tensor<double, host_memory_space> c_iter(extents[10]);
//		tensor<double, host_memory_space> c_cuv(extents[10]);
//		tensor<double, host_memory_space> c_ope(extents[10]);
//		v = matrix.shape();
//
//		for (unsigned int i = 0; i<a.shape(0); i++) {
//			a[i] = drand48();
//			b[i] = drand48();
//		}
//		for (unsigned int i = 0; i<a.shape(0); i++)
//			c_iter[i] = a[i] + b[i];
//
//
//	}
//}

//template<>
//int main() {
//	tensor<float, host_memory_space> t(extents[5][8]);
//	tensor<float, host_memory_space> tt(extents[5][8]);
//	tensor<float, host_memory_space> result(extents[5][8]);
//	for (unsigned int i = 0; i<t.shape(0)*t.shape(1); i++){
//		t[i] = i;
//		tt[i] = i;
//	}
//	apply_binary_functor(result, t, tt, BF_ADD);
//	cout<<t<<endl;
//	cout<<result<<endl;
////	for (unsigned int i = 0; i<t.shape(0); i++) {
////		for (unsigned int j = 0; j<t.shape(1); j++)
////			cout<<t(i, j)<<" ";
////		cout<<endl;
////	}
//	return 0;
//}

