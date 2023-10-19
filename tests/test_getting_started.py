#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyPLNmodels')


# ## pyPLNmodels

# We assume the data comes from a PLN model:  $ \text{counts} \sim  \mathcal P(\exp(\text{Z}))$, where $Z$ are some unknown latent variables.
# 
# 
# The goal of the package is to retrieve the latent variables $Z$ given the counts. To do so, one can instantiate a Pln or PlnPCA model, fit it and then extract the latent variables.  

# ### Import the needed functions

# In[2]:


from pyPLNmodels import get_real_count_data, get_simulated_count_data, load_model, Pln, PlnPCA, PlnPCAcollection
import matplotlib.pyplot as plt


# ### Load the data

# In[3]:


counts, labels  = get_real_count_data(return_labels=True) # np.ndarray


# ### PLN model

# In[4]:


pln = Pln(counts, add_const = True)
pln.fit()


# In[5]:


print(pln)


# #### Once fitted, we can extract multiple variables:

# In[6]:


gaussian = pln.latent_variables
print(gaussian.shape)


# In[7]:


model_param = pln.model_parameters
print(model_param["coef"].shape)
print(model_param["covariance"].shape)


# ### PlnPCA model

# In[8]:


pca = PlnPCA(counts, add_const = True, rank = 5)
pca.fit()


# In[9]:


print(pca)


# In[10]:


print(pca.latent_variables.shape)


# In[11]:


print(pca.model_parameters["components"].shape)
print(pca.model_parameters["coef"].shape)


# ### One can save the model in order to load it back after:

# In[13]:


pca.save()
dict_init = load_model("PlnPCA_nbcov_1_dim_200_rank_5")
loaded_pca = PlnPCA(counts, add_const = True, dict_initialization=  dict_init)
print(loaded_pca)


# ### One can fit multiple PCA and choose the best rank with BIC or AIC criterion

# In[14]:


pca_col = PlnPCAcollection(counts, add_const = True, ranks = [5,15,25,40,50])
pca_col.fit()


# In[15]:


pca_col.show()


# In[16]:


print(pca_col)


# ### One can extract the best model found (according to AIC or BIC criterion).

# #### AIC best model

# In[17]:


print(pca_col.best_model(criterion = "AIC"))


# #### BIC best model

# In[18]:


print(pca_col.best_model(criterion = "BIC"))


# #### Visualization of the individuals (sites) with PCA on the latent variables.

# In[19]:


pln.viz(colors=labels)
plt.show()


# In[20]:


best_pca = pca_col.best_model()
best_pca.viz(colors = labels)
plt.show()


# ### What would give a PCA on the log normalize data ? 

# In[21]:


from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns


# In[22]:


sk_pca = PCA(n_components = 2)
pca_log_counts = sk_pca.fit_transform(np.log(counts + (counts == 0)))
sns.scatterplot(x = pca_log_counts[:,0], y = pca_log_counts[:,1], hue = labels)


# ### Visualization of the variables

# In[23]:


pln.plot_pca_correlation_graph(["var_1","var_2"], indices_of_variables = [0,1])
plt.show()


# In[24]:


best_pca.plot_pca_correlation_graph(["var_1","var_2"], indices_of_variables = [0,1])
plt.show()


# ### Visualization of each components of the PCA
# 

# In[25]:


pln.scatter_pca_matrix(color = labels, n_components = 5)
plt.show()


# In[26]:


best_pca.scatter_pca_matrix(color = labels, n_components = 6)
plt.show()


# In[ ]:




