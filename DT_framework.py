import numpy as np
import random

class Block:
    def define_block(self, type, features_considered, feature, thresh, value, impurity, impurity_type, samples, depth):
        self.type = type # Node/Leaf
        self.features_considered = features_considered # features considered in split decision
        self.feature = feature # None for leaf node
        self.thresh = thresh # None for leaf node or if the splitting feature is discrete
        self.value = value # Value of the block if it's a leaf (None if the block is node)
        self.impurity = impurity # impurity in the block (entropy/variance)
        self.impurity_type = impurity_type # the type of impurity ('entropy'/'variance')
        self.samples = samples # indices of the samples that ended up in this block
        self.depth = depth # depth at which this block exists

    def generate_branches(self):
        self.left_branch = Block()
        self.right_branch = Block()
        return {'left': self.left_branch, 'right': self.right_branch}
    
    def describe_block(self):
        print(f"Block Type: {self.type}")
        print(f"At depth: {self.depth}")
        if self.type == 'Node' and self.thresh is not None: print(f"Splits at feature: {self.feature} | Thresh: {self.thresh}")
        elif self.type == 'Node' and self.thresh is None: print(f"Splits at feature: {self.feature}")
        if self.value is not None: print(f"Leaf Value: {self.value}")
        print(f"Features Considered: {self.features_considered}")
        print(f"Impurity: {self.impurity} | Impurity Type: {self.impurity_type}")
        print(f"Num Samples: {len(self.samples)}")

class DecisionTree:
    def __init__(self, X, y, task):
        # discrete features in X should be one-hot encoded
        assert task in ('regression', 'classification'), "Invalid task type"
        self.X = X
        if y.ndim == 2:
            preserve_axis = 1
            if y.shape[0] > y.shape[1]: preserve_axis = 0
            y = y.reshape(y.shape[preserve_axis])
        self.y = y
        if task=='classification': self.classes_to_predict = len(np.unique(y))
        self.task = task # classification/regression
        self.feature_type = self.determine_feature_type(self.X)

    def determine_feature_type(self, X):
        features = X.shape[1]
        feature_type = {}
        for i in range(features):
            if len(np.unique(X[:, i])) == 2 and all(np.unique(X[:, i]) == np.array([0,1])):
                feature_type[i] = 'discrete'
            else:
                feature_type[i] = 'continuous'
        return feature_type
    
    def compute_entropy(self, y):
        entropy = 0
        if len(y) != 0:
            for cat in range(self.classes_to_predict):
                p = len(y[y==cat])/len(y)
                if p != 0:
                    entropy += -p * np.log2(p) 
        #print(entropy)
        return entropy 
    
    def make_split(self, X, node_indices, feature, thresh=None):
        if self.feature_type[feature] == 'discrete':
            left_indices = node_indices[X[node_indices][:, feature] == 1]
            right_indices = node_indices[X[node_indices][:, feature] == 0]

        elif self.feature_type[feature] == 'continuous':
            assert thresh != None, f"Feature {feature} is continuous; give a suitable threshold"
            left_indices = node_indices[X[node_indices][:, feature] <= thresh]
            right_indices = node_indices[X[node_indices][:, feature] > thresh]

        return left_indices, right_indices
    
    def compute_information_gain(self, X, y, node_indices, feature, thresh=None):
        # computes information gain(if the task is classification) or
        # reduction in variance (if the task is regression)

        #the variable 'information gain' represents reduction in variance if the task is regression
        #the entropy variables represent variance if the task is regression
        #this is to avert duplicacy of code
        left_indices, right_indices = self.make_split(X, node_indices, feature, thresh)

        y_node = y[node_indices]
        y_left = y[left_indices]
        y_right = y[right_indices]

        information_gain = 0 #is reduction in variance if the task is regression

        if self.task == 'classification':
            node_entropy = self.compute_entropy(y_node)
            left_entropy = self.compute_entropy(y_left)
            right_entropy = self.compute_entropy(y_right)

        else:
            node_entropy = np.var(y_node)
            left_entropy = np.var(y_left)
            right_entropy = np.var(y_right)

        w_left = len(y_left)/len(y_node)
        w_right = len(y_right)/len(y_node)

        weighted_entropy = w_left*left_entropy + w_right*right_entropy

        information_gain = node_entropy - weighted_entropy
        return information_gain
    
    def get_best_split(self, X, y, features_to_consider, node_indices):
        max_info_gain = 0
        best_feature = -1
        best_thresh = None
        for feature in features_to_consider:
            if self.feature_type[feature] == 'discrete':
                information_gain = self.compute_information_gain(X, y, node_indices, feature)
                #print(information_gain)
                if information_gain >= max_info_gain:
                    max_info_gain = information_gain
                    best_feature = feature
            else: # feature is continuous
                possible_thresh = np.convolve(np.sort(np.unique(X[:, feature])), [0.5, 0.5], mode='valid')
                for thresh in possible_thresh:
                    information_gain = self.compute_information_gain(X, y, node_indices, feature, thresh)
                    if information_gain >= max_info_gain:
                        max_info_gain = information_gain
                        best_feature = feature
                        best_thresh = thresh

        return best_feature, best_thresh, max_info_gain
    
    def give_leaf_value(self, y):
        if self.task == 'classification': return np.argmax(np.bincount(y.astype(int))) #most common category
        else: return np.mean(y)
    
    def check_stopping_criteria(self, y, depth, information_gain,  
                                left_indices, right_indices, max_depth, min_info_gain, min_samples):
        criteria = [False, False, False, False, False]
        if max_depth is not None and depth == max_depth: criteria[0] = True
        if min_info_gain is not None and information_gain < min_info_gain: criteria[1] = True
        if min_samples is not None and len(y) <= min_samples: criteria[2] = True
        if len(np.unique(y)) == 1: criteria[3] = True
        if len(left_indices)==0 or len(right_indices)==0: criteria[4] = True

        return any(criteria)
    
    def recursive_build(self, X, y, node_indices, block, current_depth, branch_name, num_features='all',
                        max_depth=None, min_info_gain=None, min_samples=None, print_opt=True):
        if num_features == 'all': num_features = X.shape[1]
        features_to_consider = random.sample(list(range(X.shape[1])), k=num_features) #for random forest

        best_feature, best_thresh, max_info_gain = self.get_best_split(X, y,features_to_consider, node_indices)
        if best_feature != -1: left_indices, right_indices = self.make_split(X, node_indices, best_feature, best_thresh)
        else: left_indices, right_indices = [], []
        stopping_criteria_met = self.check_stopping_criteria(y[node_indices], current_depth, max_info_gain,
                                                             left_indices, right_indices,
                                                             max_depth, min_info_gain, min_samples)
        if self.task == 'classification':
            impurity_type = 'Entropy'
            impurity = self.compute_entropy(y[node_indices])
        else:
            impurity_type = 'Variance'
            impurity = np.var(y[node_indices])
        
        if stopping_criteria_met:
            formatting = " "*current_depth + "-"*current_depth
            if print_opt: print(f"{formatting} Depth {current_depth}, {branch_name} leaf with indices {node_indices}")

            leaf_value = self.give_leaf_value(y[node_indices])
            block.define_block(type='Leaf', features_considered=sorted(features_to_consider),
                               feature=None, thresh=None, value=leaf_value, impurity=impurity, 
                               impurity_type=impurity_type, samples=node_indices, depth=current_depth)
            
        else:
            formatting = "-"*current_depth
            if print_opt: print(f"{formatting} Depth {current_depth},{branch_name}: Split on feature: {best_feature}")
            block.define_block(type='Node', features_considered=sorted(features_to_consider),
                               feature=best_feature, thresh=best_thresh, value=None, impurity=impurity, 
                               impurity_type=impurity_type, samples=y[node_indices], depth=current_depth)
            branches = block.generate_branches()
            #left_indices, right_indices = self.make_split(X, node_indices, best_feature, best_thresh)
            self.recursive_build(X, y, left_indices, branches['left'], current_depth+1, 'Left', num_features,
                                 max_depth, min_info_gain, min_samples, print_opt)
            self.recursive_build(X, y, right_indices, branches['right'], current_depth+1, 'Right', num_features,
                                 max_depth, min_info_gain, min_samples, print_opt)
            
    def build_tree(self, max_depth=None, min_info_gain=None, min_sample=None, print_opt=True):
        self.tree = Block()
        root_indices = np.array(range(self.X.shape[0]))
        self.recursive_build(self.X, self.y, root_indices, self.tree, 0, 'Root', 'all',
                             max_depth, min_info_gain, min_sample, print_opt)
        
    def run_inference(self, X, block, node_indices=None, y=None, inferences_done=None):
        #block --> instance of Block class
        if X.ndim == 1: X = X.reshape((1, len(X)))
        if y is None: y = np.zeros(X.shape[0])
        if node_indices is None: node_indices = np.array(range(X.shape[0]))
        if inferences_done is None: inferences_done = y.astype(bool)

        if block.type == 'Node':
            left_indices, right_indices = self.make_split(X, node_indices, block.feature, block.thresh)

            y_left = self.run_inference(X, block.left_branch, left_indices, y, inferences_done)
            y_right = self.run_inference(X, block.right_branch, right_indices, y, inferences_done)
            if y_left is not None: return y_left
            if y_right is not None: return y_right
        elif block.type == 'Leaf':
            y[node_indices] = block.value
            inferences_done[node_indices] = True

            if all(inferences_done):
                return y
            
    def measure_error(self, tree, X, y):
        y_pred = self.run_inference(X, tree)
        if self.task == 'classification':
            error = np.sum(y_pred != y)/len(y)
        else:
            error = (1/len(y))*np.sum((y-y_pred)**2)
        return error
    
class RandomForest(DecisionTree):
    def bootstrap_sampling(self, X, y):
        m = X.shape[0]
        sample_indices = random.choices(list(range(m)), k=m)
        return X[sample_indices], y[sample_indices]
    
    def build_ensemble(self, num_trees, num_features, max_depth=None, min_info_gain=None, min_sample=None, print_opt=True):
        # set num_features for bagging
        if num_features == 'all': num_features=self.X.shape[1]
        self.ensemble = {}
        root_indices = np.array(range(self.X.shape[0]))
        for i in range(1, num_trees+1):
            if print_opt: print(f"Building Tree {i}.......")
            tree = Block()
            X, y = self.bootstrap_sampling(self.X, self.y)
            self.recursive_build(X, y, root_indices, tree, 0, 'Root', num_features,
                        max_depth, min_info_gain, min_sample, print_opt=False)
            self.ensemble[i] = tree

    def most_common_value(self, row):
        return np.argmax(np.bincount(row.astype(int)))
    
    def run_inference_on_ensemble(self, X, ensemble):
        # ensemble: dictionary consisting trees (instances of Block class)
        if X.ndim == 1: X = X.reshape((1, len(X)))
        y = np.zeros((X.shape[0], len(ensemble)))
        at_tree = 0
        for tree_id in ensemble:
            y[:, at_tree] = self.run_inference(X, ensemble[tree_id])
            at_tree += 1

        if self.task == 'classification':
            return np.apply_along_axis(self.most_common_value, axis=1, arr=y)
        else:
            return np.mean(y, axis=1)
        
    def measure_error(self, ensemble, X, y):
        y_pred = self.run_inference_on_ensemble(X, ensemble)
        if self.task == 'classification':
            error = np.sum(y_pred != y)/len(y)
        else:
            error = (1/len(y))*np.sum((y-y_pred)**2)
        return error
    
class AdaBoostClassifier(DecisionTree):
    def __init__(self, X, y):
        super().__init__(X, y, 'classification')

    def initialise_weights(self, X):
        return np.ones(X.shape[0])/X.shape[0]
    
    def compute_total_error(self, y_pred, y, sample_weights):
        return np.sum(sample_weights[y_pred != y])/np.sum(sample_weights)
    
    def calculate_say(self, total_error):
        if total_error != 1 and total_error !=0: 
            return 0.5*np.log((1-total_error)*(self.classes_to_predict-1)/total_error)
        elif total_error == 1: return 0.5*np.log(1e-08)
        elif total_error == 0: return 0.5*np.log((self.classes_to_predict-1)/1e-08)

    def assign_sample_weights(self, y_pred, y, sample_weights, say, learning_rate):
        sample_weights[y!=y_pred] *= np.exp(say*learning_rate)
        sample_weights[y==y_pred] *= np.exp(-say*learning_rate)
        sample_weights = sample_weights/np.sum(sample_weights)
        return sample_weights
    
    def make_new_dataset(self, X, y, sample_weights):
        bins = np.cumsum(sample_weights)
        new_indices = np.digitize(np.random.rand(X.shape[0]), bins=bins)
        return X[new_indices], y[new_indices]
    
    def train(self, n_learners, depth=1, learning_rate=1, store_error=False, X_cv=None, y_cv=None):
        X = self.X.copy()
        y = self.y.copy()
        sample_weights = self.initialise_weights(X)
        say = []
        learners = []

        self.train_error_hist = np.zeros(n_learners)
        self.cv_error_hist = np.zeros(n_learners)
        self.total_error_hist = np.zeros(n_learners)
        for i in range(n_learners):
            print(f"Building learner: {i+1}....")
            weak_learner = Block()
            self.recursive_build(X, y, np.array(range(X.shape[0])), weak_learner, 0, 'root',
                                 'all', depth, print_opt=False)
            y_pred = self.run_inference(X, weak_learner)
            total_error = self.compute_total_error(y_pred, y, sample_weights)
            self.total_error_hist[i] = total_error
            say_t = self.calculate_say(total_error)
            say.append(say_t)
            sample_weights = self.assign_sample_weights(y_pred, y, sample_weights, say_t, learning_rate)
            X, y = self.make_new_dataset(X, y, sample_weights)
            sample_weights = self.initialise_weights(X)
            learners.append(weak_learner)

            if store_error:
                self.train_error_hist[i] = self.measure_error([learners, np.array(say)], self.X, self.y)
                if X_cv is not None and y_cv is not None:
                    self.cv_error_hist[i] = self.measure_error([learners, np.array(say)], X_cv, y_cv)
        self.learners = [learners, say]

    def predict(self, X, learners_say):
        if X.ndim == 1: X = X.reshape((1, len(X)))
        learners = learners_say[0]
        say = learners_say[1]
        y_pred = np.zeros((self.classes_to_predict, X.shape[0]))
        for i in range(len(learners)):
            y_curr = self.run_inference(X, learners[i])
            say_mask = np.zeros_like(y_pred)
            say_mask[y_curr.astype(int), np.arange(len(y_curr))] = 1
            y_pred = y_pred + say_mask*say[i]
        y_pred = np.argmax(y_pred, axis=0)
        return y_pred
    
    def measure_error(self, learners_say, X, y):
        y_pred = self.predict(X, learners_say)
        return np.sum(y_pred != y)/len(y)
    
class AdaBoostRegressor(DecisionTree):
    '''Implementation of AdaBoost.R2 Algorithm'''
    def __init__(self, X, y):
        super().__init__(X, y, 'regression')

    def initialise_weights(self, X):
        return np.ones(X.shape[0])/X.shape[0]
    
    def compute_losses(self, y_pred, y):
        residual = np.abs(y-y_pred)
        losses = residual/np.max(residual)
        return losses
    
    def compute_model_error(self, losses, sample_weights):
        return np.sum(losses*sample_weights)
    
    def calculate_beta(self, model_error):
        if model_error != 1 and model_error != 0:
            return model_error/(1-model_error)
        elif model_error == 1:
            return 1e08
        elif model_error == 0:
            return 1e-08

    def assign_sample_weights(self, losses, sample_weights, beta):
        sample_weights = (beta**(1-losses))*sample_weights
        sample_weights = sample_weights/np.sum(sample_weights)
        return sample_weights
        

    def make_new_dataset(self, X, y, sample_weights):
        bins = np.cumsum(sample_weights)
        new_indices = np.digitize(np.random.rand(X.shape[0]), bins=bins)
        return X[new_indices], y[new_indices]
    
    def train(self, n_learners, depth=1, store_error=False, X_cv=None, y_cv=None):
        sample_weights = self.initialise_weights(self.X)
        beta = []
        learners = []

        self.train_error_hist = []
        self.cv_error_hist = []
        self.model_error_hist = []
        
        for t in range(1, n_learners+1):
            print(f"Building learner: {t}...")
            X_sampled, y_sampled = self.make_new_dataset(self.X, self.y, sample_weights)
            weak_learner = Block()
            self.recursive_build(X_sampled, y_sampled, np.array(range(X_sampled.shape[0])), 
                                 weak_learner, 0, 'root', 'all', depth, print_opt=False)
            y_pred = self.run_inference(self.X, weak_learner)
            losses = self.compute_losses(y_pred, self.y)
            model_error = self.compute_model_error(losses, sample_weights)

            if model_error >= 0.5:
                print("    "+"-> Model error exceeded 0.5")
                print(f"    -> Total learners generated: {t-1}")
                break
            self.model_error_hist.append(model_error)
            learners.append(weak_learner)
            beta_t = self.calculate_beta(model_error)
            beta.append(beta_t)
            sample_weights = self.assign_sample_weights(losses, sample_weights, beta_t)

            if store_error:
                self.train_error_hist.append(self.measure_error([learners, np.array(beta)], self.X, self.y))
                if X_cv is not None and y_cv is not None:
                    self.cv_error_hist.append(self.measure_error([learners, np.array(beta)], X_cv, y_cv))
        self.learners = [learners, np.array(beta)]



    def compute_weighted_median(self, values, weights):
        weights = weights/np.sum(weights)
        ascending_indices = np.argsort(values)
        values = values[ascending_indices]
        weights = weights[ascending_indices]
        return values[np.searchsorted(np.cumsum(weights), 0.5)]

    def predict(self, X, learners_beta):
        if X.ndim == 1: X = X.reshape((1, len(X)))
        learners = learners_beta[0]
        weights = np.log(1/learners_beta[1])
        all_pred = np.zeros((len(learners), X.shape[0]))
        for i in range(len(learners)):
            all_pred[i, :] = self.run_inference(X, learners[i])
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = self.compute_weighted_median(all_pred[:, i], weights.copy())
        return y_pred
    
    def measure_error(self, learners_beta, X, y):
        # returns mean squared error
        y_pred = self.predict(X, learners_beta)
        return (1/len(y))*np.sum((y-y_pred)**2)
    
class GradientBoostRegressor(DecisionTree):
    def __init__(self, X, y):
        super().__init__(X, y, 'regression')

    def predict(self, X, trees):
        y_pred = self.run_inference(X, trees[0])
        for tree in trees[1:]:
            y_pred = y_pred + self.learning_rate*self.run_inference(X, tree)
        return y_pred

    def compute_residuals(self, X, y, trees):
        return y - self.predict(X, trees)

    def compute_error(self, X, y, trees, error_type=2):
        '''
        error_type 1: Mean difference
        error_type 2: MSE'''
        assert error_type in (1,2), "Invalid error type"
        if error_type==1:
            return (1/len(y))*np.sum(np.abs(self.compute_residuals(X, y, trees)))
        else:
            return (1/len(y))*np.sum((y-self.predict(X, trees))**2)
    
    def train(self, learning_rate, n_trees, max_depth, min_mean_difference=None, store_hist=False, 
              X_cv=None, y_cv=None, error_type=2):
        X = self.X
        y = self.y
        self.learning_rate = learning_rate
        initial_leaf = Block()
        initial_leaf.define_block('Leaf', list(range(X.shape[1])), None, None, 
                                  self.give_leaf_value(y), np.var(y), 'variance', y, 0)
        self.trees = [initial_leaf]
        self.train_error_hist = []
        self.cv_error_hist = []

        for i in range(n_trees):
            residuals = self.compute_residuals(X, y, self.trees)
            error = self.compute_error(X, y, self.trees, error_type)

            if store_hist:
                self.train_error_hist.append(error)
                if X_cv is not None and y_cv is not None:
                    self.cv_error_hist.append(self.compute_error(X_cv, y_cv, self.trees, error_type))

            if min_mean_difference is not None and error_type==1:
                if error < min_mean_difference:
                    print(f"    -> Mean difference ({error}) is less than the threshold ({min_mean_difference})")
                    print(f"    -> Trees generated: {len(self.trees)-1}")
                    break
            print(f"Building tree: {i+1}....")
            tree = Block()
            self.recursive_build(X, residuals, np.array(range(X.shape[0])), tree, 0, 'root',
                                max_depth=max_depth, print_opt=False)
            self.trees.append(tree)


class GradientBoostClassifier(DecisionTree):
    THRESH = 0.5
    def __init__(self, X, y):
        super().__init__(X, y, 'classification')

    def initial_leaf_value(self, y):
        return np.log(np.sum((y==1).astype(int)) / np.sum((y==0).astype(int)))
    
    def log_odds_prediction(self, X, trees):
        y_pred = self.run_inference(X, trees[0])
        for tree in trees[1:]:
            y_pred = y_pred + self.learning_rate*self.run_inference(X, tree)
        return y_pred
    
    def log_odds_to_probability(self, log_odds_pred):
        return np.exp(log_odds_pred)/(1 + np.exp(log_odds_pred))
    
    def predict(self, X, trees, enable_thresh=False):
        y_pred = self.log_odds_to_probability(self.log_odds_prediction(X, trees))
        if enable_thresh:
            y_pred = (y_pred > self.THRESH).astype(int)
        return y_pred
    
    def give_leaf_value(self, indices, residuals, prev_probabilities):
        return np.sum(residuals[indices])/np.sum(prev_probabilities[indices]*(1-prev_probabilities[indices]))

    def recursive_build(self, X, y, node_indices, block, current_depth, prev_probabilities, branch_name, 
                        num_features='all', max_depth=None, min_info_gain=None, min_samples=None, 
                        print_opt=True):
        # for GradientBoostClassifier, y represents the residuals
        if num_features == 'all': num_features = X.shape[1]
        features_to_consider = random.sample(list(range(X.shape[1])), k=num_features) #for random forest

        best_feature, best_thresh, max_info_gain = self.get_best_split(X, y,features_to_consider, node_indices)
        if best_feature != -1: left_indices, right_indices = self.make_split(X, node_indices, best_feature, best_thresh)
        else: left_indices, right_indices = [], []
        stopping_criteria_met = self.check_stopping_criteria(y[node_indices], current_depth, max_info_gain,
                                                             left_indices, right_indices,
                                                             max_depth, min_info_gain, min_samples)

        impurity_type = 'Variance'
        impurity = np.var(y[node_indices])
        
        if stopping_criteria_met:
            formatting = " "*current_depth + "-"*current_depth
            if print_opt: print(f"{formatting} Depth {current_depth}, {branch_name} leaf with indices {node_indices}")

            leaf_value = self.give_leaf_value(node_indices, y, prev_probabilities)
            block.define_block(type='Leaf', features_considered=sorted(features_to_consider),
                               feature=None, thresh=None, value=leaf_value, impurity=impurity, 
                               impurity_type=impurity_type, samples=y[node_indices], depth=current_depth)
            
        else:
            formatting = "-"*current_depth
            if print_opt: print(f"{formatting} Depth {current_depth},{branch_name}: Split on feature: {best_feature}")
            block.define_block(type='Node', features_considered=sorted(features_to_consider),
                               feature=best_feature, thresh=best_thresh, value=None, impurity=impurity, 
                               impurity_type=impurity_type, samples=y[node_indices], depth=current_depth)
            branches = block.generate_branches()
            #left_indices, right_indices = self.make_split(X, node_indices, best_feature, best_thresh)
            self.recursive_build(X, y, left_indices, branches['left'], current_depth+1, prev_probabilities,
                                 'Left', num_features, max_depth, min_info_gain, min_samples, print_opt)
            self.recursive_build(X, y, right_indices, branches['right'], current_depth+1, prev_probabilities, 
                                 'Right', num_features, max_depth, min_info_gain, min_samples, print_opt)
            
    def measure_error(self, X, y, trees, error_type=1):
        '''
        error_type 1: mean difference between actual and predicted probabilites
        error_type 2: fraction of misclassified examples
        '''
        assert error_type in (1,2), "Invalid error type"
        pred_probabilities = self.predict(X, trees)
        if error_type == 1: 
            return (1/len(y))*(np.sum(np.abs(y-pred_probabilities)))
        else: 
            predictions = (pred_probabilities>self.THRESH).astype(int)
            return np.sum((y!=predictions).astype(int))/len(y)
            
    def train(self, learning_rate, n_trees, max_depth, min_mean_difference=None, store_hist=False,
              X_cv=None, y_cv=None, error_type=1):
        self.learning_rate = learning_rate
        X = self.X
        y = self.y
        initial_leaf = Block()
        initial_leaf.define_block('Leaf', list(range(X.shape[1])), None, None,
                                  self.initial_leaf_value(y), self.compute_entropy(y), 'entropy', y, 0)
        self.trees = [initial_leaf]
        self.train_error_hist = []
        self.cv_error_hist = []
        self.task = 'regression'
        # though the actual task is classification, we are building trees with continuous values (residuals)
        for i in range(n_trees):
            probabilities = self.predict(X, self.trees)
            residuals = y - probabilities
            error = self.measure_error(X, y, self.trees, error_type)

            if store_hist:
                self.train_error_hist.append(error)
                if X_cv is not None and y_cv is not None:
                    self.cv_error_hist.append(self.measure_error(X_cv, y_cv, self.trees, error_type))
        
            if min_mean_difference is not None and error==1:
                if error < min_mean_difference:
                    print(f"    -> Mean difference ({error}) is less than the threshold ({min_mean_difference})")
                    print(f"    -> Trees generated: {len(self.trees)-1}")
                    break 
            print(f"Buidling tree: {i+1}....")
            tree = Block()
            self.recursive_build(X, residuals, np.array(range(X.shape[0])), tree, 0, probabilities, 'root',
                                 max_depth=max_depth, print_opt=False)
            self.trees.append(tree)