#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <vector>
#include <random>
using ll = long long;
using ld = long double;
using Graph = std::vector<std::vector<int>>;

class KMeans { 
public:
    KMeans(int dimention,int classSize, int dataNum,int seed){
        Dimention = dimention;
        Class_Size = classSize;
        Max_Patt = dataNum;
        Seed = seed;

        InitCenter.resize(classSize,(std::vector<double>(Dimention)));
        Data.resize(Max_Patt,std::vector<double>(Dimention));
        Center.resize(classSize,std::vector<double>(Dimention));
        Sum.resize(classSize,std::vector<double>(Dimention));
        ClassNum.resize(dataNum);
        ElemNum.resize(classSize);
        pointDistance.resize(classSize);
    }
    void insertData();//入力用メソッド(テストよう)
    void insertData(std::vector<std::vector<double>> data);//入力
    void dataInit();//本来はランダムに設定するが、今回はめんどいのでこれでいく
    double Distance(std::vector<double> p1, std::vector<double> p2);
    int MinClass(std::vector<double> p);
    double Random(int a);
    //tol 収束範囲
    std::vector<int> fit(double tol);

private:
    int Dimention;
    int Class_Size;
    int Max_Patt;
    int Seed;
    std::vector<std::vector<double>> Data;
    std::vector<std::vector<double>> InitCenter;
    std::vector<std::vector<double>> Center;//各クラスタ中心位置
    std::vector<std::vector<double>> Sum;//合計値
    std::vector<int> ClassNum;//分類パターンの属するクラスタ番号
    std::vector<int> ElemNum;//クラスタに含まれる要素数
    std::vector<double> pointDistance;//クラスタの中心位置と分類パターンの距離
};

void KMeans::insertData(){
    for(int i = 0; i < Max_Patt; ++i) {
        for(int j = 0; j < Dimention; ++j) {
           std::cin >> Data[i][j]; 
           std::cout << Data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void KMeans::insertData(std::vector<std::vector<double>> data){
    for(int i = 0; i < (int)data.size(); ++i) {
        for(int j = 0; j < (int)data[i].size(); ++j) {
            Data[i][j] = data[i][j];
        }
    }
}

//テスト用の代替処理 本来は乱数等でやる
void KMeans::dataInit() {
    for(int i = 0; i < Class_Size; ++i) {
        for(int j = 0; j < Dimention; ++j) {
            Center[i][j] = Random(Seed);
            InitCenter[i][j] = Random(Seed);
            std::cout << Center[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

double KMeans::Random(int a) {
    std::random_device rnd; // 非決定的な乱数生成
    std::mt19937 mt(rnd()); // 擬似乱数生成（メルセンヌ・ツイスター）。上で生成した乱数をシード値として扱う
    std::uniform_int_distribution<int> intRand(0,a);
    return (double)intRand(mt)/(double)100;
}


double KMeans::Distance(std::vector<double> p1,std::vector<double> p2) {
    double res = 0.0;
    for(int i = 0; i < Dimention; ++i) {
        res += pow(p1[i]-p2[i],2);
    }
    return sqrt(res);
}

int KMeans::MinClass(std::vector<double> p){ 
    double Dist = p[0];
    int MinNum = -1;
    for(int i = 0; i < Class_Size; ++i){
        if(Dist >= p[i]) {
            MinNum = i;
            Dist = p[i];
        }
    }
    std::cout << "MinClass :" << Dist << std::endl; 
    return MinNum;
}




std::vector<int> KMeans::fit(double tol){
    while(true) {
        for(int i = 0; i < Class_Size; ++i) {
            for(int j = 0; j < Dimention; ++j) {
                Sum[i][j] = 0.0;
            }
            ElemNum[i] = 0;
        }

        for(int i = 0; i < Max_Patt; ++i) {
            for(int j = 0; j < Class_Size; ++j) {
                pointDistance[j] = Distance(Center[j],Data[i]);
            }
            ClassNum[i] = MinClass(pointDistance);
        }
        
        //中心位置修正
        for(int i = 0; i < Max_Patt; ++i) {
            if((ClassNum[i] >= 0) && (ClassNum[i] < Class_Size)) {
                ElemNum[ClassNum[i]]++;//バケツ
                for(int j = 0; j < Dimention; ++j) {
                    Sum[ClassNum[i]][j] += Data[i][j];
                }
            }
        }
        for(int i = 0; i < Class_Size; ++i) {
            std::cout << ElemNum[i] << std::endl;
            for(int j = 0; j < Dimention; ++j) {
                if(ElemNum[i]){
                    //重心計算
                    Center[i][j] = Sum[i][j]/(double)ElemNum[i];
                    std::cout << Sum[i][j];
                }
            }
        }
        
        std::cout << std::endl;
        bool EndFlag = true;
        for(int i = 0; i < Class_Size; ++i) {
            if(Distance(Center[i],InitCenter[i]) > tol) EndFlag=false;
        }
        if(EndFlag){
            break;
        } else {
            for(int i = 0; i < Class_Size; ++i) {
                for(int j = 0; j < Dimention; ++j) {
                    InitCenter[i][j] = Center[i][j];
                } 
            }
        }
    }
    return ClassNum;
}


std::vector<int> KMeansPy(int dimention,int classSize, int dataNum ,std::vector<std::vector<double>> data,double tol,int seed) {
    auto *km = new KMeans(dimention,classSize,dataNum,seed);
    km->dataInit();
    km->insertData(data);
    std::vector<int> res = km->fit(tol);
    return res;
}

PYBIND11_MODULE(MyModule, m) {
  m.def("KMeans", &KMeansPy);
}
/* 
int main() {
    //次元数　クラスタ数　データ数
    auto *km = new KMeans(2,4,8);
    km->dataInit();//今回は、時間短縮のためこう書いている   
    km->insertData();//データの追加
    //K-means法実行
    std::vector<int> ans = km->fit(0.1);
    //分類結果
    for(auto a : ans) {
        std::cout << a << std::endl;
    }
    
    delete km;

    return 0;
} */