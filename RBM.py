import numpy as np
class RBM(object):
    '''
        object:
        1. numOfHidden: The hidden layer's number.
        2. numOfVisible: The visible layer's number.
        3. hidden_bias
        4. visible_bias
    '''
    def __init__(self,hiddenNum,visibleNum,k,lr,momentum=0.5,weight_decay=1e-4):
        self.k=k
        self.lr=lr
        self.momentum=momentum
        self.weight_decay=weight_decay
        self.numOfHidden=hiddenNum
        self.numOfVisible=visibleNum
    
        # Initialize the random weights.
        # The weight should be zero-mean with standard deviate 0.01 gaussian.
        # Visible bias: log(p[i]/1-p[i])
        # hidden bias: 0
        seed=np.random.RandomState(20240701)
        self.weights=seed.normal(0,0.01,size=(self.numOfVisible,self.numOfHidden))
        self.hiddenBias=np.zeros(self.numOfHidden)
        self.visibleBias=np.zeros(self.numOfVisible)
    
        self.weights_momemtum=np.zeros(self.numOfHidden,self.numOfVisible)
        self.visible_bias_momentem=np.zeros(self.numOfVisible)
        self.hidden_bias_momentem=np.zeros(self.numOfHidden)
        
    def C_D(self,train_data): # The Contrastive Divergence Algorithm.
        # positive phase
        pos_visible_probs=train_data
        pos_hidden_probs=self._func(np.matmul(pos_visible_probs,self.weights)+self.hiddenBias)
        pos_hidden_activations=(pos_hidden_probs >= self._random_prob(self.numOfHidden)).astype(np.float)
        pos_expectations=np.matmul(pos_visible_probs.T,pos_hidden_activations)
        
        # negative phase
        hidden_activations=pos_hidden_activations
        
        for i in range(self.k):
            visible_probs=self._func(np.matmul(hidden_activations,self.weights)+self.visibleBias)
            hidden_probs=self._func(np.matmul(visible_probs,self.weights)+self.hiddenBias)
            hidden_activations=(hidden_probs >= self._random_prob(self.numOfHidden)).astype(np.float)
        
        neg_visible_probs=visible_probs
        neg_hidden_probs=hidden_probs
        neg_expectations=np.matmul(neg_visible_probs.T,neg_hidden_probs)
        
        # Update parameters
        self.weights_momemtum *= self.momentum
        self.weights_momemtum += (pos_expectations-neg_expectations)
        
        self.visible_bias_momentem *= self.momentum
        self.visible_bias_momentem += np.sum(pos_visible_probs-neg_visible_probs,axis=0)
        
        self.hidden_bias_momentem *= self.momentum
        self.hidden_bias_momentem += np.sum(pos_hidden_probs-neg_hidden_probs,axis=0)
        
        batch_size=train_data.shape[0]
        self.weights += self.weights_momemtum * self.lr / batch_size
        self.visibleBias += self.visible_bias_momentem * self.lr / batch_size
        self.hiddenBias += self.hidden_bias_momentem * self.lr / batch_size
        
        # weight decay L2;
        self.weights -= self.weights * self.weight_decay
        
        # reconstruct error
        return np.sum((train_data-neg_visible_probs)**2)
    
    def _func(self,x):
        return 1.0/(1.0+np.exp(-x))
    
    def _random_prob(self, num):
        return np.random.random(num)