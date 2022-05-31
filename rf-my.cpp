#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <cmath>
 
using namespace std;
//乱数生成
class RandXor{
private:
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
public:
    RandXor(){
        init();
    }
    void init(){
        x=123456789;
		y=362436069;
		z=521288629;
		w= 88675123; 
    }

    inline unsigned int random() {
		unsigned int t;
		t=(x^(x<<11));x=y;y=z;z=w; return( w=(w^(w>>19))^(t^(t>>8)) );
	}

    inline void randomShuffle(vector<int> &a){
        const int n = (int)a.size();
        for(int i = n-1; i > 0; --i) {
            swap(a[i],a[random()%(i+1)]);
        }
    }
};

static RandXor randxor;//プログラム開始から終了まで記憶し続ける変数

struct TreeNode {
    bool leaf;//葉(=子がない)ならば、true
    int level;//ノードの深さ。ルートは0
    int featureID;//説明変数ID.x0,x1,x2,x3
    double value;//分割する値
    double answer;//ノード内(=領域内)の目的変数yの平均値
    vector<int> bags;//ノード内（＝領域内）に含まれるデータのID
    int left;//左側の子のノードID
    int right;//右側の子のノードID
    
    TreeNode() {
        leaf = false;
        level = -1;
        featureID = -1;
        value = 0;
        answer = 0;
        left = -1;
        right = -1;
    }
};

class DecisionTree {
private:
    //nodeを葉にして、curNodeを次のノードへ進める
    void setLeaf(TreeNode& node,int& curNode, const vector<double>& answer){
        node.leaf = true;
        //回帰の場合は,目的変数yの平均を求める。　
        //(node.bags)ノード内に含まれるデータIDでループ
        for(int p : node.bags){
            node.answer += answer[p];
        }
        assert(node.bags.size() > 0);
        if(node.bags.size()) { 
            node.answer /= node.bags.size();
        }
        curNode++;
    }
    vector<TreeNode> m_nodes; //決定木のノードたち。m_nodes[0]がルート
public:
    DecisionTree(){}
    /* 学習　訓練データをいれて、決定木を作成。
     *features 説明変数x0,x1,x2,x3,x4
     *answer 目的変数y
     * minNodeSize ノード内
     * maxLevel ノードの深さの最大値
     * numRandomFeatures 領域を分ける時に試す説明変数の数
     * numRandomPositions 領域を分ける時に試すデータの数
     */
    DecisionTree(const vector<vector<double>>& features,const vector<double>& answers, int minNodeSize,
    int maxLevel,int numRandomFeatures, int numRandoomPositions){
        const int numData = features.size();//説明変数のラベル数
        const int numFeatures = features[0].size();//それぞれの説明変数の個数
        assert(numData == answers.size());//説明変数の個数と目的変数の個数が同じ
        assert(numData>1);//説明変数の個数は、１より大きいことを想定
        TreeNode root;
        root.bags.resize(numData);//ルートのデータ個数を説明変数の個数でresize
        for(int i = 0; i < numData; ++i) { 
            //同一IDになる可能性もあるが、よい
            root.bags[i] = randxor.random()%numData;
        }
        m_nodes.emplace_back(root);

        int curNode = 0;//現在のノード番号？？
        //m_nodesに子ノードが追加されていく。BFS
        while(curNode < m_nodes.size()) {
            TreeNode &node = m_nodes[curNode];//今のノードを代入
            //現在ノードに入っている目的変数が、すべて同じかどうかを調べる(すべておなじなら分ける意味なくなる)
            bool equal = true;//すべて同じならtrue
            for(int i = 1; i < (int)node.bags.size(); ++i) {
                if(answers[node.bags[i]] != answers[node.bags[i-1]]) {//ノード内の要素を探索
                    equal = false;
                    break;
                }
            }
            //葉になる条件のチェック  node.levelノードの深さ
            if(equal || (int)node.bags.size() <= minNodeSize || node.level >= maxLevel) {
                //葉にして子ノードは増やさない
                setLeaf(node,curNode,answers);
                continue;
            }

            //どこで分けるのがベストかを調べる
            int bestFeatureID = -1;
            int bestLeft=0,bestRight=0;
            double bestValue =0,bestMSE = 1e99;//平均との二乗の差の和
            //説明変数の個数でループ　
            for(int i = 0; i < numRandomFeatures; ++i) { 
                //x0,x1,x2,...　の、どの軸で分けるかを決める それぞれの説明変数の個数numFeatures
                //featureIDは、分割する値のID？
                const int featureID = randxor.random()%numFeatures;
                //領域を分ける時に試すデータの数でループ
                for(int j = 0; j < numRandoomPositions;++j) {//どの位置で分けるのかを決める
                    const int positionID = randxor.random()%(int)node.bags.size();
                    const double splitValue = features[node.bags[positionID]][featureID];

                    double sumLeft = 0,sumRight = 0;
                    int totalLeft = 0;//分割する値未満の個数
                    int totalRight = 0;//分割する値以上の個数
                    for(int p : node.bags) {
                        if(features[p][featureID] < splitValue) {
                            sumLeft += answers[p];
                            totalLeft++;
                        } else {
                            sumRight += answers[p];
                            totalRight++;
                        }
                    }

                    //どちらかが０ということは、分ける意味がない（分類されていない）
                    if(totalLeft == 0 || totalRight == 0) continue;

                    //平均との差を求める　回帰するので必要
                    double meanLeft = totalLeft == 0 ? 0 : sumLeft/totalLeft;
                    double meanRight = totalRight == 0 ? 0 : sumRight/totalRight;

                    double mse = 0; //平均二乗誤差　1/nΣ(y_pred - y_t)^2
                    for (int p : node.bags){
                        if(features[p][featureID] < splitValue){
                            mse += pow((answers[p] - meanLeft),2);
                        } else {
                            mse += pow((answers[p] - meanRight),2);                           
                        }
                    }
                    //誤差関数の値が小さければ更新する。
                    if(mse < bestMSE) {
                        bestMSE = mse;
                        bestValue = splitValue;
                        bestFeatureID = featureID;
                        bestLeft = totalLeft;
                        bestRight = totalRight;
                    }
                }
            }

            //どちらか片方に偏った場合、葉にする
            if(bestLeft==0 || bestRight==0) {
                setLeaf(node,curNode,answers);
                continue;
            }

            //うまく分けられたので、新しい子ノードを左右に追加
            TreeNode left;
            TreeNode right;

            left.level = right.level = node.level+1;
            node.featureID = bestFeatureID;//説明変数のID
            node.value = bestValue;//分割する値
            node.left = (int)m_nodes.size();//左のノードID
            node.right = (int)m_nodes.size()+1;

            left.bags.resize(bestLeft);//左側に分割された値の個数でメモリ確保
            right.bags.resize(bestRight);
            //IDを左側と右側に割当、分割する値(bestValue)で説明変数により判断
            for(int p : node.bags) {
                if(features[p][node.featureID] < node.value){
                    left.bags[--bestLeft] = p;
                } else {
                    right.bags[--bestRight] = p;
                }
            }
            m_nodes.emplace_back(left);
            m_nodes.emplace_back(right);
            curNode++;
        }
    }
    //予測
    //features テスト用の説明変数x0,x1,x2のセット
    //返り値　目的変数ｙの予測値
    double estimate(const vector<double>& features) const {
        //ルートからたどる
        //葉にたどり着くまで探索を続ける事で値がでる
        const TreeNode *pNode = &m_nodes[0];
        while(true){
            if(pNode->leaf) {//子がない　つまり葉に達した時
                break;
            }
            //分割する値よりも小さければ左側の子のノードIDを持ってくる　二分探索木ってやつなんか？
            const int nextNodeID = features[pNode->featureID] < pNode->value ? pNode->left : pNode->right;
            pNode = &m_nodes[nextNodeID];
        }
        return pNode->answer;
    }
};

class RandomForest{
private:
    vector<DecisionTree> m_tree; //決定木の配列
public:
    RandomForest(){
        clear();
    }
    void clear() {
        m_tree.clear();
    }
    //訓練
    //features　説明変数のセット
    //answer 目的変数ｙのセット
    //treeNo 追加する木の数
    //minNodeSize ノード内
    void train(const vector<vector<double>>& features, const vector<double>& answers,int treeNo, int minNodeSize) {
        for(int i = 0; i < treeNo; ++i) {
            m_tree.emplace_back(DecisionTree(features,answers,minNodeSize,16,2,5));
        }
    }

    //回帰の予測 
    //features 説明変数
    double estimateRegression(vector<double> &features){
        if((int)m_tree.size() == 0) {
            return 0.0;
        }
        //全ての木から得られた結果の平均を取るだけ
        double sum = 0;
        for(int i = 0; i < (int)m_tree.size(); ++i) {
            sum += m_tree[i].estimate(features);
        }
        return sum/(int)m_tree.size();
    }
};

int main() {
    int numAll,numTrainings,numTests,numFeatures;
    //全データ数　訓練データ数　テストデータ数　説明変数の数
    cin >> numAll >> numTrainings >> numTests >> numFeatures;
    
    assert(numTrainings+numTests<=numAll);
    // 全データ
	vector<vector<double>> allFeatures(numAll, vector<double>(numFeatures));
	vector<double> allAnswers(numAll);

    for(int i = 0; i < numAll; ++i) {
        for(int j = 0; j < numFeatures; ++j) {
            cin >> allFeatures[i][j];
        }
        cin >> allAnswers[i];
    }
    //シャッフル
    vector<int> shuffleTable;
    for(int i = 0; i < numTrainings+numTests; ++i) {
        shuffleTable.emplace_back(i);
    }
    randxor.randomShuffle(shuffleTable);

    //訓練データ データをそれぞれシャッフル
    vector<vector<double>> trainingFeatures(numTrainings,vector<double>(numFeatures));
    vector<double> trainingAnswers(numTrainings);
    for(int i = 0; i < numTrainings; ++i) { 
        trainingFeatures[i] = allFeatures[shuffleTable[i]];
        trainingAnswers[i] = allAnswers[shuffleTable[i]];
    }
    //テストデータ
    vector<vector<double>> testFeatures(numTests,vector<double>(numFeatures));
    vector<double> testAnswers(numTests);
    for(int i = 0; i < numTests; ++i) {
        testFeatures[i] = allFeatures[shuffleTable[numTrainings+i]];
        testAnswers[i] = allAnswers[shuffleTable[numTrainings+i]];
    }

    RandomForest* rf = new RandomForest();
    //rf->train(trainingFeatures, trainingAnswers, 20, 1);
	// 木を徐々に増やしていく
	int numTrees = 0;
	for (int k = 0; k < 20; ++k)
	{
		// 学習
		const int numAdditionalTrees = 1;
		rf->train(trainingFeatures, trainingAnswers, numAdditionalTrees, 1);
		numTrees += numAdditionalTrees;
 
		// 予測と結果表示
		cout << "-----" << endl;
		cout << "numTrees=" << numTrees << endl;
		double totalError = 0.0;
		for (int i = 0; i < numTests; ++i)
		{
			const double myAnswer = rf->estimateRegression(testFeatures[i]);
			const double diff = myAnswer-testAnswers[i];
			totalError += abs(diff);
			cout << " myAnswer=" << myAnswer << " testAnswer=" << testAnswers[i] << " diff=" << diff << endl;
		}
		cout << "totalError=" << totalError << endl;
	}
    delete rf;

    return 0;
}
