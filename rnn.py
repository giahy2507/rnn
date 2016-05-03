__author__ = 'HyNguyen'

import numpy as np
import activation as act
import utils as u



class Softmaxlayer(object):
    def __init__(self, rng, input, n_in, n_out):
        self.input = input
        self.W = rng.normal(size=(n_in,n_out))
        self.b = rng.normal(size=(1,n_out))
        self.output = softmax(np.dot(self.input,self.W) + self.b)
        self.pred = self.output
        self.gradient = None

    def compute_gradient(self, t):
        self.gradient = self.pred - t

class InternalNone(object):
    def __init__(self, label, p, p_unnormalize, left_child, right_child):
        self.label = label
        self.p = p
        self.p_unnormalize = p_unnormalize
        self.left_child = left_child
        self.right_child = right_child
        self.child_num = left_child.child_num + right_child.child_num
        self.gradient = None

class LeafNone(object):
    def __init__(self, label, p):
        self.label = label
        self.p = p.reshape(1,-1)
        self.p_unnormalize = p
        self.child_num = 1

class RecursiveNeuralNetworl(object):
    def __init__(self, act_func = act.sigmoid, embsize = 50, mnb_size = 100, lr = 0.1, l2_reg_level = 0.001, wordvector = None, name ="RecursiveNeuralNetwork"):

        # init wordvector
        self.wordvector = wordvector
        self.embsize = embsize

        # activation function
        self.act_func = act_func

        # output dimension
        self.output_dim = 2 # sentiment 0 1

        # params
        self.mnb_size = mnb_size
        self.lr = lr
        self.l2_reg_level = l2_reg_level
        self.name = name

        self.rng = np.random.RandomState(4488)

        # init weight hidden layer
        self.Wh_l = u.init_w(self.rng,(self.embsize, self.embsize))
        self.Wh_r = u.init_w(self.rng,(self.embsize, self.embsize))
        self.bh   = u.init_b((1,self.embsize))

        # init weight softmax layer
        self.Ws = u.init_w(self.rng,(self.embsize, self.output_dim))
        self.bs = u.init_b((1,self.output_dim))

        # Gradients
        self.dWh_l = np.empty(self.Wh_l.shape)
        self.dWh_r = np.empty(self.Wh_r.shape)
        self.dbh = np.empty(self.bh.shape)
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty(self.bs.shape)

    def set_params(self, params):
        """
        Params
            params: is tuple (self.Wh_l, self.Wh_r, self.bh, self.Ws, self.bs)
        """
        if len(params) == 5:
            self.Wh_l, self.Wh_r, self.bh, self.Ws, self.bs  = params

    def forward_tree(self, root):
        if len(root) == 1:
            if isinstance(root[0],unicode):
                label = str(root[0])
                return LeafNone(label, wordvector.wordvector(label))
            elif len(root[0]) == 1:
                if isinstance(root[0][0],unicode):
                    label = str(root[0][0])
                    return LeafNone(label, wordvector.wordvector(label))
                elif len(root[0][0]) == 1:
                    if isinstance(root[0][0][0], unicode):
                        label = str(root[0][0][0])
                        return LeafNone(label, wordvector.wordvector(label))

        left_node = self.forward_tree(root[0])
        right_node = self.forward_tree(root[1])

        left_p = left_node.p
        right_p = right_node.p
        p_unnormalize = self.act_func.activate(np.dot(left_p,self.Wh_l)+ np.dot(right_p, self.Wh_r) + self.bh)
        # p = p_unnormalize/(np.linalg.norm(p_unnormalize))
        p = p_unnormalize
        return InternalNone(left_node.label + " | " + right_node.label, p, p_unnormalize ,left_node, right_node)

    def forward_softmax(self, root):
        Z = np.dot(root.p, self.Ws) + self.bs
        A = act.softmax(Z)
        return A

    def forward(self, root, label):
        """
        Param
            root: root of NLTK Tree
        Returns
            root_node: InternalNode of binary tree
            softmax_layer: softmax layer
        """
        root_node = self.forward_tree(root)
        softmax_layer = self.forward_softmax(root_node)
        pred = np.argmax(softmax_layer)
        cost = -np.log(softmax_layer[0,label])
        return root_node, softmax_layer, cost, pred

    def backward_softmax(self, tree, softmax_layer, label):
        # one host label
        t = np.zeros((1,self.output_dim), dtype=np.float32)
        t[:,label] = 1

        # back propagation softmax
        delta = softmax_layer - t
        grad = np.dot(tree.p.T,delta) + self.l2_reg_level*self.Ws
        self.dWs += grad
        self.dbs += delta
        # return gradient propagation for previous layer
        back_grad = np.dot(delta, self.Ws.T)
        return back_grad

    def backward_tree(self, tree, back_grad):
        if isinstance(tree, LeafNone):
            return
        # delta + grand for left side and right side
        delta_left = back_grad*self.act_func.derivative(tree.left_child.p)
        grad_left = np.dot(tree.left_child.p.T,delta_left) + self.l2_reg_level*self.Wh_l
        delta_right = back_grad*self.act_func.derivative(tree.right_child.p)
        grad_right = np.dot(tree.right_child.p.T,delta_right) + self.l2_reg_level*self.Wh_r
        self.dbh = self.dbh + delta_left + delta_right
        self.dWh_l += grad_left
        self.dWh_r += grad_right
        back_grad_left = np.dot(delta_left, self.Wh_l.T)
        back_grad_right = np.dot(delta_right, self.Wh_r.T)
        self.backward_tree(tree.left_child, back_grad_left)
        self.backward_tree(tree.right_child, back_grad_right)


    def backward(self, tree, softmax_layer, label):
        """
        Back propagation
        Params:
            tree: root node
            softmax_layer: softmax layer of NN
            label: true label
        Returns:
            xxx
        """
        back_grad = self.backward_softmax(tree,softmax_layer,label)
        self.backward_tree(tree,back_grad)

    def cost_grad(self, mnb_trees, mnb_correct = [] ,test=False):
        cost = 0.0

        mnb_predict = []
        mnb_softmax_layer = []
        mnb_roots_node = []

        # Zero gradients
        self.dWh_l[:] = 0
        self.dWh_r[:] = 0
        self.dbh[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0

        for i,tree in enumerate(mnb_trees):
            root_node,softmax_layer,_cost,pred = self.forward(tree,mnb_correct[i])
            mnb_softmax_layer.append(softmax_layer)
            mnb_roots_node.append(root_node)
            cost += _cost
            mnb_predict.append(pred)

        if test is True:
            return (1./ len(mnb_trees)) * cost, mnb_correct, mnb_predict

        # backward
        for i,tree in enumerate(mnb_trees):
            self.backward(mnb_roots_node[i],mnb_softmax_layer[i],mnb_correct[i])

        avg = 1. / self.mnb_size

        cost += (self.l2_reg_level/2)*np.sum(self.Wh_l**2)
        cost += (self.l2_reg_level/2)*np.sum(self.Wh_r**2)
        cost += (self.l2_reg_level/2)*np.sum(self.Ws**2)
        cost = avg*cost

        gWh_l = avg * self.dWh_l
        gWh_r = avg * self.dWh_r
        gbh = avg * self.dbh
        gWs = avg * self.dWs
        gbs = avg * self.dbs

        return cost, gWh_l, gWh_r, gbh, gWs, gbs


    def check_grad(self, data, label = [] , epsilon= 1e-6):

        cost, gWh_l, gWh_r, gbh, gWs, gbs = self.cost_grad(data,label)
        err = 0.0
        count = 0.0
        print 'Checking dWh_l...'
        Wh_l = self.Wh_l[...,None]
        dWh_l = gWh_l
        for i in xrange(Wh_l.shape[0]):
            for j in xrange(Wh_l.shape[1]):
                Wh_l[i,j] += epsilon
                costP,_,_,_,_,_ = self.cost_grad(data,label)
                Wh_l[i,j] -= epsilon
                grad = (costP - cost) / epsilon
                err += np.abs(dWh_l[i, j] - grad)
                count+=1

        if 0.001 > err/count:
            print "Grad check passed for dWh"
        else:
            print "Grad check failed for dWh: sum of error = %.9f"%(err/count)



from nltk.parse.stanford import StanfordParser
from nltk.treetransforms import chomsky_normal_form
from nltk.tree import Tree
from vector.wordvectors import WordVectors
parser = StanfordParser(path_to_jar="/Users/HyNguyen/Downloads/stanford-parser-full-2015-12-09/stanford-parser.jar",
                        path_to_models_jar="/Users/HyNguyen/Downloads/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar"
                        ,model_path="/Users/HyNguyen/Downloads/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

if __name__ == "__main__":

    rng = np.random.RandomState(4488)
    wordvector = WordVectors.load_from_text_format("model/word2vec.txt", "word2vec")
    pos_sent = []
    neg_sent = []
    with open("data/rt-polarity.neg.txt",mode="r") as f:
        neg_sent.append(f.readline())
        neg_sent.append(f.readline())
        neg_sent.append(f.readline())

    with open("data/rt-polarity.pos.txt",mode="r") as f:
        pos_sent.append(f.readline())
        pos_sent.append(f.readline())
        pos_sent.append(f.readline())


    trees = []
    labels = [0]*3 + [1]*3
    sents = pos_sent + neg_sent
    for sent in sents:
        a = list(parser.raw_parse(sent))
        hytree = a[0]
        chomsky_normal_form(hytree)
        trees.append(hytree[0])

    rnn = RecursiveNeuralNetworl(embsize=300,mnb_size=6,wordvector=wordvector)

    trees[0].pretty_print()

    for tree,label in zip(trees,labels):
        root_node, softmax_layer, cost, pred = rnn.forward(tree,label)
        print("correct {0}, predict {1}, cost {2}".format(label,pred,cost))




    # if have_1_child
        # child_have_1_child_and_unicode
    # if have_2_child