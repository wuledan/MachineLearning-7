/************************************************************************/
/* ���ڱ�Ҷ˹���ı�����													*/
/* Keiko@20140823														*/
/************************************************************************/
#include <math.h>
#include <algorithm>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;

class NaiveBayes
{
public:
	void train(string file_path);//ѵ��
	void classfiy(string file_path);//����
private:
	double getProbality(vector<wstring> &attr,int &label);//���ݲ��������������������label�¿��ܵĸ���
	void readTrainingData(string file_path,vector<int>* classes,vector<vector<wstring>> *matrix);//��ȡѵ�����Ͳ��Լ�
private:
	vector<int> m_classesTraining;//�洢ѵ�������������
	vector<vector<wstring>> m_matrixTraining;//�洢ѵ����������
	vector<int> m_classesClassify;//�洢���������������
	vector<vector<wstring>> m_matrixClassify;//�洢������������
	map<int,map<wstring,double>*> m_attrcooure;//����(ÿ��Ϊһ���ı�����)
	map<int,int> m_unique_class_attr_counts;//�����,��������ά����
	int m_totalAttrCount;//�ܵ�������������
	map<int,double> m_probCi;//����������ĳ�������ֵĸ���P(Ci)
};
const wchar_t delim=' ';

void NaiveBayes::train(string file_path)
{
	readTrainingData(file_path,&m_classesTraining,&m_matrixTraining);
	//����P(Ci)
	map<int,double> countClass;//key:����ʶ;value:ѵ�������и�������������
	for (vector<int>::iterator iter_vi=m_classesTraining.begin();iter_vi!=m_classesTraining.end();++iter_vi)
	{
		if (countClass.find(*iter_vi)==countClass.end())
		{
			countClass[*iter_vi]=1.0;
		}
		else
		{
			countClass[*iter_vi]++;
		}
	}
	
	for (map<int,double>::iterator iter_mid=countClass.begin();iter_mid!=countClass.end();iter_mid++)
	{
		m_probCi[iter_mid->first]=(double)iter_mid->second/(double)m_classesTraining.size();
	}
	//P(Xi|Cj)=Count(Xi,Cj)/Count(Cj);�������ڸ�������г��ֵĸ���=�����Cj�г���Xi���ĵ���/���Cj���ĵ���
	m_totalAttrCount=0;
	map<wstring,int> attrCount;
	for (unsigned int i=0;i<m_matrixTraining.size();++i)
	{
		vector<wstring> wstr=m_matrixTraining[i];
		map<wstring,int> cur_map;//��¼��ǰ�����дʳ��ֵĴ���
		for (vector<wstring>::iterator iter_vw=wstr.begin();iter_vw!=wstr.end();iter_vw++)
		{
			cur_map[*iter_vw]=0;
		}
		vector<wstring>::iterator iter_vw;
		for (iter_vw=wstr.begin();iter_vw!=wstr.end();iter_vw++)
		{
			if (m_attrcooure.find(m_classesTraining[i])==m_attrcooure.end())
			{
				m_attrcooure[m_classesTraining[i]]=new map<wstring,double>;
				(*m_attrcooure[m_classesTraining[i]])[*iter_vw]=1;
				cur_map[*iter_vw]=1;
			}
			else
			{
				map<wstring,double> *exist_map=m_attrcooure[m_classesTraining[i]];
				//����ôʲ�������Ci�����У������ô�
				if ((*exist_map).find(*iter_vw)==(*exist_map).end())
				{
					(*exist_map)[*iter_vw]=1;
				}
				//����ô��Ѿ�������Ci��������ֻ����������ֵĴ���
				if (cur_map[*iter_vw]==0)
				{
					(*exist_map)[*iter_vw]++;
					cur_map[*iter_vw]=1;
				}
			}
			if (attrCount.find(*iter_vw)==attrCount.end())
			{
				attrCount[*iter_vw];
				m_totalAttrCount++;
			}
		}
	}
	cout<<"��������ά��"<<m_totalAttrCount<<endl;
	map<int,map<wstring,double>*> class_attr_pairs;
	for (unsigned int i=0;i<m_matrixTraining.size();i++)
	{
		if (class_attr_pairs.find(m_classesTraining[i])==class_attr_pairs.end())
		{
			class_attr_pairs[m_classesTraining[i]]=new map<wstring,double>;
		}
		map<wstring,double>* tmp=class_attr_pairs[m_classesTraining[i]];
		vector<wstring> wv=m_matrixTraining[i];
		for (vector<wstring>::iterator iter_vw=wv.begin();iter_vw!=wv.end();iter_vw++)
		{
			if ((*tmp).find(*iter_vw)==(*tmp).end())
			{
				(*tmp)[*iter_vw]=1;
			}
		}
	}
	for (map<int,map<wstring,double>*>::iterator iter_mimwd=class_attr_pairs.begin();iter_mimwd!=class_attr_pairs.end();++iter_mimwd)
	{
		map<wstring,double>* tmp=iter_mimwd->second;
		double nCount=0;
		for (map<wstring,double>::iterator iter_mwd=(*tmp).begin();iter_mwd!=(*tmp).end();iter_mwd++)
		{
			nCount+=iter_mwd->second;
		}
		m_unique_class_attr_counts[iter_mimwd->first]=(int)nCount;
		cout<<"���"<<iter_mimwd->first<<"����������������"<<nCount<<endl;
	}
}
void NaiveBayes::readTrainingData(string file_path,vector<int>* classes,vector<vector<wstring>> *matrix)
{
	locale china("chs");//ʹ������ 
	wcin.imbue(china); 
	wcout.imbue(china);  
	wstring s;  
	wchar_t wc=L' ';// L"���ַ�"  
	std::wifstream file(file_path);
	std::wstring line;
	FILE* fp = fopen(file_path.c_str(), "rt+,ccs=UTF-8");  
	wchar_t temp[4096]={'\0'};
	int nCount=0;
	while (!feof(fp))
	{
		//cout<<"���ڶ�ȡ��"<<++nCount<<"������"<<endl;
		fgetws(temp,4096,fp);//ÿ�ζ�һ��
		wstring wstr=wstring(temp);
		wistringstream iss(wstr);
		wstring attr;
		vector<wstring> vw;
		int classname;
		bool flag=false;
		while (getline(iss,attr,delim))
		{
			if (!flag)
			{
				classname=stoi(attr);//���洢�ڵ�һ��
				flag=true;
			}
			else
			{
				vw.push_back(attr);
			}
		}
		classes->push_back(classname);
		matrix->push_back(vw);
	}
}
double NaiveBayes::getProbality(vector<wstring> &attr,int &label)
{
	//����P(X|Ci)
	double log_prob_x_given_ci=0;//����Ci,P(X)�ĸ��ʣ�P(X|Ci)
	for (unsigned int i=0;i<attr.size();i++)
	{
		wstring ws=attr[i];
		double count_xi_ci=0;//(Xi,Ci)���ֲ����N
		map<wstring,double>* tmp=m_attrcooure[label];//ȡ��label������������
		map<wstring,double>::iterator iter_wd=tmp->find(ws);
		if (iter_wd!=tmp->end())
		{
			count_xi_ci=iter_wd->second;//����ҵ���ѳ��ִ�������count_xi_ci
		}
		double temp=(double)m_unique_class_attr_counts[label];//label����������������
		double prob_xi_given_ci=(double)(count_xi_ci+1)/(double)(temp+m_totalAttrCount);
		log_prob_x_given_ci +=log(prob_xi_given_ci);
	}
	return (log(m_probCi[label])+log_prob_x_given_ci);
}

void NaiveBayes::classfiy(string file_path)
{
	readTrainingData(file_path,&m_classesClassify,&m_matrixClassify);
	vector<vector<wstring>>::iterator iter_vvw;
	vector<bool> accuracy;//���ڴ洢�Ƿ��ж���ȷ
	for (unsigned int i=0;i<m_matrixClassify.size();i++)
	{
		vector<wstring> vw=m_matrixClassify[i];
		set<int> label;
		vector<int>::iterator iter_vi;
		int max_class;
		double max_prob=-DBL_MAX;
		for (iter_vi=m_classesTraining.begin();iter_vi!=m_classesTraining.end();iter_vi++)
		{
			if (label.find(*iter_vi)==label.end())
			{
				double prob=getProbality(vw,(*iter_vi));
				if (prob>max_prob)
				{
					max_prob=prob;
					max_class=*iter_vi;
				}
				label.insert(*iter_vi);//�����ж�����Ƿ��Ѵ���
			}
		}
		//ͳ��׼ȷ��
		(m_classesClassify[i]==max_class)?accuracy.push_back(true):accuracy.push_back(false);
	}
	double acc=0;
	for (vector<bool>::iterator iter=accuracy.begin();iter!=accuracy.end();iter++)
	{
		if (*iter==true)
		{
			acc++;
		}
	}
	acc/=(double)accuracy.size();
	cout<<"������ȷ��Ϊ:"<<acc<<"%";
}

int main(int argc,char *argv[])
{
	NaiveBayes naivebayes;
	string training_file_path="train.dat";
	string test_file_path="test.dat";
	naivebayes.train(training_file_path);
	naivebayes.classfiy(test_file_path);
}