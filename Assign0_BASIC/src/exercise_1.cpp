/*
 * exercise_1.cpp
 *
 *  Created on: Apr 10, 2014
 *      Author: Kostadin Cvejoski
 */
#include <iostream>
#include <cuv.hpp>
#include <typeinfo>

using namespace std;
using namespace cuv;

template<typename T, typename H, typename D>
tensor<T, H> add(tensor<T, H>& a, tensor<T, D>& b);

template<typename T, typename M>
tensor<T, M> add_for(tensor<T, M>& a, tensor<T, M>& b);

template<typename T, typename M>
tensor<T, M> add_apply(tensor<T, M>& a, tensor<T, M>& b);

template<typename T, typename H, typename D>
bool differen_types(tensor<T, H> a, tensor<T, D> b);

template<typename T, typename M>
void rand_init(tensor<T, M>& a);

template<typename T, typename M>
void print_3D_view(tensor<T, M>& a);

template<typename T, typename M>
void set_value(tensor_view<T, M>& a, T v);

template<typename T, typename M>
void set_value(tensor<T, M>& a, T v);

void task_0();
void task_1();
void task_2();
void task_2_1();
void task_2_2();
void task_2_3();
void task_3();

#define BOOST_TEST_MODULE CuvTest
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE(TASK_0)
{
	{
		task_0();
	}
}

BOOST_AUTO_TEST_CASE(TASK_1)
{
	{
		task_1();
	}
}

BOOST_AUTO_TEST_CASE(TASK_2)
{
	{
		task_2();
	}
}

BOOST_AUTO_TEST_CASE(TASK_3)
{
	{
		task_3();
	}
}

template<typename T, typename H, typename D>
tensor<T, H> add(tensor<T, H>& a, tensor<T, D>& b) {
	tensor<T, H> result;
	tensor<T, H> i = b;
	result = a + i;
	return result;
}

template<typename T, typename M>
tensor<T, M> add_for(tensor<T, M>& a, tensor<T, M>& b) {
	tensor<T, M> result(extents[a.size()]);
	for (unsigned int i=0; i<a.size(); i++)
		result[i] = a[i] + b[i];
	return result;
}

template<typename T, typename M>
tensor<T, M> add_apply(tensor<T, M>& a, tensor<T, M>& b) {
	tensor<T, M> result(extents[a.size()]);
	apply_binary_functor(result, a, b, BF_ADD);
	return result;
}

template<typename T, typename H, typename D>
bool differen_types(tensor<T, H> a, tensor<T, D> b){
	string d = typeid(a).name();
	if (d.compare(typeid(b).name()))
		return true;
	return false;
}

template<typename T, typename M>
void rand_init(tensor<T, M>& a){
	for (unsigned int i=0; i<a.shape(0);i++)
		a[i] = drand48();
}

template<typename T, typename M>
void print_2D_view(tensor<T, M>& a) {
	for (unsigned int i = 0; i<a.shape(0); i++){
		for (unsigned int j = 0; j<a.shape(1); j++)
			cout<<a(i, j)<<" ";
		cout<<endl;
	}
}

template<typename T, typename M>
void print_3D_view(tensor<T, M>& a) {
	for (unsigned int k = 0; k<a.shape(0); k++) {
		cout<<"dimension 0: "<<k<<endl;
		for (unsigned int i = 0; i<a.shape(1); i++){
			for (unsigned int j = 0; j<a.shape(2); j++)
				cout<<a(k, i, j)<<" ";
			cout<<endl;
		}
	}
}

template<typename T, typename M>
void set_value(tensor_view<T, M>& a, T v) {
	a = v;
	//for (unsigned int i = 0; i<a.size(); i++)
	//	a[i] = v;
}

template<typename T, typename M>
void set_value(tensor<T, M>& a, T v) {
	a = v;
	//for (unsigned int i = 0; i<a.size(); i++)
	//	a[i] = v;
}

void task_0() {
	const int n = 4;

	tensor<float, host_memory_space> a_h(extents[n]);
	tensor<float, host_memory_space> b_h(extents[n]);

	tensor<float, host_memory_space> c0_h(extents[n]);
	tensor<float, host_memory_space> c1_h(extents[n]);
	tensor<float, host_memory_space> c2_h(extents[n]);

	tensor<float, dev_memory_space> a_d(extents[n]);
	tensor<float, dev_memory_space> b_d(extents[n]);

	tensor<float, dev_memory_space> c0_d(extents[n]);
	tensor<float, dev_memory_space> c1_d(extents[n]);
	tensor<float, dev_memory_space> c2_d(extents[n]);

	tensor<float, host_memory_space> c0_dc(extents[n]);
	tensor<float, host_memory_space> c1_dc(extents[n]);
	tensor<float, host_memory_space> c2_dc(extents[n]);

	rand_init(a_h);
	rand_init(b_h);

	a_d = a_h.copy();
	b_d = b_h.copy();

	c0_h = add_for(a_h, b_h);
	c1_h = add_apply(a_h, b_h);
	c2_h = add(a_h, b_h);

	c0_d = add_for(a_d, b_d);
	c1_d = add_apply(a_d, b_d);
	c2_d = add(a_d, b_d);

	c0_dc = c0_d.copy();
	c1_dc = c1_d.copy();
	c2_dc = c2_d.copy();

	BOOST_CHECK_EQUAL_COLLECTIONS(c0_h.ptr(), &c0_h[c0_h.size()-1], c1_h.ptr(), &c1_h[c1_h.size()-1]);
	BOOST_CHECK_EQUAL_COLLECTIONS(c1_h.ptr(), &c1_h[c0_h.size()-1], c2_h.ptr(), &c2_h[c2_h.size()-1]);

	BOOST_CHECK_EQUAL_COLLECTIONS(c0_dc.ptr(), &c0_dc[c0_dc.size()-1], c1_dc.ptr(), &c1_dc[c1_dc.size()-1]);
	BOOST_CHECK_EQUAL_COLLECTIONS(c1_dc.ptr(), &c1_dc[c1_dc.size()-1], c2_dc.ptr(), &c2_dc[c2_dc.size()-1]);

	BOOST_CHECK_EQUAL_COLLECTIONS(c0_h.ptr(), &c0_h[c0_h.size()-1], c0_dc.ptr(), &c0_dc[c0_dc.size()-1]);
	BOOST_CHECK_EQUAL_COLLECTIONS(c1_h.ptr(), &c1_h[c1_h.size()-1], c1_dc.ptr(), &c1_dc[c1_dc.size()-1]);
	BOOST_CHECK_EQUAL_COLLECTIONS(c2_h.ptr(), &c2_h[c2_h.size()-1], c2_dc.ptr(), &c2_dc[c2_dc.size()-1]);

}

void task_1() {
	tensor<float, host_memory_space> a_host(extents[3][8]);
	tensor<float, host_memory_space> b_host(extents[3][8]);
	tensor<float, host_memory_space> r_host(extents[3][8]);

	tensor<float, dev_memory_space> a_dev(extents[3][8]);
	tensor<float, dev_memory_space> b_dev(extents[3][8]);

	//result from the dev_memory_space we copy to the dev_memory_space
	//because BOOST_CHECK_EQUAL_COLLECTIONS cannot access dev_memory_space
	tensor<float, host_memory_space> r_dev(extents[3][8]);

	for (unsigned int i = 0; i<a_dev.shape(0)*a_dev.shape(1); i++){
		a_dev[i] = i*10;
		a_host[i] = i*10;
		b_dev[i] = i*100;
		b_host[i] = i*100;
	}

	r_host = add(a_host, b_host);
	r_dev = add(a_dev, b_dev);
	BOOST_CHECK_EQUAL_COLLECTIONS(r_host.ptr(), &r_host[r_host.size()-1], r_dev.ptr(), &r_dev[r_dev.size()-1]);
}

void task_2() {
	//task_2_1();
	//task_2_2();
	task_2_3();
}

void task_2_1(){
	tensor<float, host_memory_space> t(extents[10][11]);
	for (unsigned int i = 0; i<t.size(); i++)
		t[i] = 0.f;
	tensor_view<float, host_memory_space> tv1 = t[indices[index_range(1,3)]];
	cout<<"original tensor\n";
	for (unsigned int i = 0; i<t.shape(0); i++) {
		for (unsigned int j = 0; j<t.shape(1); j++)
			cout<<t(i, j)<<" ";
			cout<<endl;
	}

	sequence(tv1);
	cout<<"after sequence tensor\n";
	for (unsigned int i = 0; i<t.shape(0); i++) {
		for (unsigned int j = 0; j<t.shape(1); j++)
			cout<<t(i, j)<<" ";
			cout<<endl;
	}
}

void task_2_2() {
	tensor<float, host_memory_space> t(extents[10][11]);
	for (unsigned int i = 0; i<t.size(); i++)
		t[i] = 0.f;
	tensor_view<float, host_memory_space> v = t[indices[index_range(1,3)]];
	tensor<float, host_memory_space> w = v.copy();
	cout<<"original tensor\n";
	for (unsigned int i = 0; i<t.shape(0); i++) {
		for (unsigned int j = 0; j<t.shape(1); j++)
			cout<<t(i, j)<<" ";
			cout<<endl;
	}

	sequence(w);
	cout<<"after sequence tensor\n";
	for (unsigned int i = 0; i<t.shape(0); i++) {
		for (unsigned int j = 0; j<t.shape(1); j++)
			cout<<t(i, j)<<" ";
			cout<<endl;
	}

	v = w;

	cout<<"after assignment tensor\n";
		for (unsigned int i = 0; i<t.shape(0); i++) {
			for (unsigned int j = 0; j<t.shape(1); j++)
				cout<<t(i, j)<<" ";
				cout<<endl;
		}
}

void task_2_3() {
	tensor<float, host_memory_space> t(extents[10][11][12]);
	for (unsigned int i = 0; i<t.size(); i++)
		t[i] = 0.f;

	tensor_view<float, host_memory_space> v0 = t[indices[index_range(4,7)]];
	tensor_view<float, host_memory_space> v1 = t[indices[7]];
	tensor_view<float, host_memory_space> v2 = t[indices[index_range()][index_range(4, 7)]];
	tensor_view<float, host_memory_space> v3 = t[indices[index_range()][7][index_range()]];
	tensor_view<float, host_memory_space> v4 = t[indices[index_range()][index_range()][index_range(4, 7)]];
	tensor_view<float, host_memory_space> v5 = t[indices[index_range()][index_range()][index_range(7, 8)]];

	//tensor<float, host_memory_space> w2 = v2.copy();
	tensor<float, host_memory_space> w3 = v3.copy();
	tensor<float, host_memory_space> w4 = v4.copy();
	tensor<float, host_memory_space> w5 = v5.copy();

	set_value(v0, 1.f);
	set_value(v1, 2.f);
//	set_value(w2, 3.f);
//	t[indices[index_range()][index_range(4, 7)][index_range()]] = w2;

	set_value(w3, 4.f);
	t[indices[index_range()][7][index_range()]] = w3;
	set_value(w4, 5.f);
	t[indices[index_range()][index_range()][index_range(4, 7)]] = w4;

	set_value(w5, 6.f);
	t[indices[index_range()][index_range()][index_range(7, 8)]] = w5;

	cout<<"v0 applied sequence:"<<endl;
	print_3D_view(t);
}

void task_3() {
	const unsigned int N = 200;
	const unsigned int K = 500;
	const unsigned int M = 400;

	tensor<float, host_memory_space> a(extents[N][K]);
	tensor<float, host_memory_space> b(extents[K][M]);
	tensor<float, host_memory_space> c(extents[N][M]);
	tensor<float, host_memory_space> c1(extents[N][M]);
	tensor<float, host_memory_space> d(extents[M]);
	tensor<float, host_memory_space> d1(extents[M]);
	tensor<float, host_memory_space> e(extents[N]);
	tensor<float, host_memory_space> x(extents[N][M]);
	tensor<float, host_memory_space> x1(extents[N][M]);

	d = 0.f;
	d1 = 0.f;
	x1 = 0.f;
	//when we use drand48() there is a problem with precision !!!

	for (unsigned int i = 0; i<a.size(); i++)
		a[i] = drand48();
	for (unsigned int i = 0; i<b.size(); i++)
		b[i] = drand48();
	for (unsigned int i = 0; i<c.size(); i++)
		c[i] = drand48();
	for (unsigned int i = 0; i<e.size(); i++)
		e[i] = drand48();

	for (unsigned int j = 0; j<M; j++)
		for (unsigned int i = 0; i<N; i++) {
			for (unsigned int k = 0; k<K; k++)
				d1[j] += a(i, k)*b(k, j);
			d1[j] += c(i, j) + e[i];
		}
//
//	for (unsigned int i = 0; i<N; i++)
//		for (unsigned int k=0; k<M; k++)
//			for (unsigned int j = 0; j<K; j++)
//				x1(i,k) += a(i, j)*b(j, k);
////
//	for (unsigned int j = 0; j<M; j++)
//		for (unsigned int i = 0; i<N; i++)
//			c1(i, j) = c(i, j) + e[i];
////
////	for (unsigned int j = 0; j<M; j++)
////			for (unsigned int i = 0; i<N; i++) {
////				d1[j] += x1(i, j);
////				d1[j] += c1(i, j);
////			}


	prod(x, a, b);
	matrix_plus_col(c, e);
	d = sum(x, 0);
	d += sum(c, 0);

	//BOOST_CHECK_EQUAL_COLLECTIONS(x.ptr(), &x[x.size()-1], x1.ptr(), &x1[x1.size()-1]);
	//BOOST_CHECK_EQUAL_COLLECTIONS(d.ptr(), d.ptr()+d.size(), d1.ptr(), d1.ptr()+d1.size());
	//BOOST_CHECK_EQUAL_COLLECTIONS(c.ptr(), &c[c.size()-1], c1.ptr(), &c1[c1.size()-1]);
}
