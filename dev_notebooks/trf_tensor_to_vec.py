# Script adding a method to squash transformer model tensor embeddings into vectors  

# Import the Language object under the 'language' module in spaCy,
# and NumPy for calculating cosine similarity.
from spacy.language import Language
import numpy as np

# We use the @ character to register the following Class definition
# with spaCy under the name 'tensor2attr'.
@Language.factory('tensor2attr')

# We begin by declaring the class name: Tensor2Attr. The name is 
# declared using 'class', followed by the name and a colon.
class Tensor2Attr:
    
    # We continue by defining the first method of the class, 
    # __init__(), which is called when this class is used for 
    # creating a Python object. Custom components in spaCy 
    # require passing two variables to the __init__() method:
    # 'name' and 'nlp'. The variable 'self' refers to any
    # object created using this class!
    def __init__(self, name, nlp):
        
        # We do not really do anything with this class, so we
        # simply move on using 'pass' when the object is created.
        pass

    # The __call__() method is called whenever some other object
    # is passed to an object representing this class. Since we know
    # that the class is a part of the spaCy pipeline, we already know
    # that it will receive Doc objects from the preceding layers.
    # We use the variable 'doc' to refer to any object received.
    def __call__(self, doc):
        
        # When an object is received, the class will instantly pass
        # the object forward to the 'add_attributes' method. The
        # reference to self informs Python that the method belongs
        # to this class.
        self.add_attributes(doc)
        
        # After the 'add_attributes' method finishes, the __call__
        # method returns the object.
        return doc
    
    # Next, we define the 'add_attributes' method that will modify
    # the incoming Doc object by calling a series of methods.
    def add_attributes(self, doc):
        
        # spaCy Doc objects have an attribute named 'user_hooks',
        # which allows customising the default attributes of a 
        # Doc object, such as 'vector'. We use the 'user_hooks'
        # attribute to replace the attribute 'vector' with the 
        # Transformer output, which is retrieved using the 
        # 'doc_tensor' method defined below.
        doc.user_hooks['vector'] = self.doc_tensor
        
        # We then perform the same for both Spans and Tokens that
        # are contained within the Doc object.
        doc.user_span_hooks['vector'] = self.span_tensor
        doc.user_token_hooks['vector'] = self.token_tensor
        
        # We also replace the 'similarity' method, because the 
        # default 'similarity' method looks at the default 'vector'
        # attribute, which is empty! We must first replace the
        # vectors using the 'user_hooks' attribute.
        doc.user_hooks['similarity'] = self.get_similarity
        doc.user_span_hooks['similarity'] = self.get_similarity
        doc.user_token_hooks['similarity'] = self.get_similarity
    
    # Define a method that takes a Doc object as input and returns 
    # Transformer output for the entire Doc.
    def doc_tensor(self, doc):
        
        # Return Transformer output for the entire Doc. As noted
        # above, this is the last item under the attribute 'tensor'.
        # Average the output along axis 0 to handle batched outputs.
        return doc._.trf_data.tensors[-1].mean(axis=0)
    
    # Define a method that takes a Span as input and returns the Transformer 
    # output.
    def span_tensor(self, span):
        
        # Get alignment information for Span. This is achieved by using
        # the 'doc' attribute of Span that refers to the Doc that contains
        # this Span. We then use the 'start' and 'end' attributes of a Span
        # to retrieve the alignment information. Finally, we flatten the
        # resulting array to use it for indexing.
        tensor_ix = span.doc._.trf_data.align[span.start: span.end].data.flatten()
        
        # Fetch Transformer output shape from the final dimension of the output.
        # We do this here to maintain compatibility with different Transformers,
        # which may output tensors of different shape.
        out_dim = span.doc._.trf_data.tensors[0].shape[-1]
        
        # Get Token tensors under tensors[0]. Reshape batched outputs so that
        # each "row" in the matrix corresponds to a single token. This is needed
        # for matching alignment information under 'tensor_ix' to the Transformer
        # output.
        tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
        
        # Average vectors along axis 0 ("columns"). This yields a 768-dimensional
        # vector for each spaCy Span.
        return tensor.mean(axis=0)
    
    # Define a function that takes a Token as input and returns the Transformer
    # output.
    def token_tensor(self, token):
        
        # Get alignment information for Token; flatten array for indexing.
        # Again, we use the 'doc' attribute of a Token to get the parent Doc,
        # which contains the Transformer output.
        tensor_ix = token.doc._.trf_data.align[token.i].data.flatten()
        
        # Fetch Transformer output shape from the final dimension of the output.
        # We do this here to maintain compatibility with different Transformers,
        # which may output tensors of different shape.
        out_dim = token.doc._.trf_data.tensors[0].shape[-1]
        
        # Get Token tensors under tensors[0]. Reshape batched outputs so that
        # each "row" in the matrix corresponds to a single token. This is needed
        # for matching alignment information under 'tensor_ix' to the Transformer
        # output.
        tensor = token.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]

        # Average vectors along axis 0 (columns). This yields a 768-dimensional
        # vector for each spaCy Token.
        return tensor.mean(axis=0)
    
    # Define a function for calculating cosine similarity between vectors
    def get_similarity(self, doc1, doc2):
        
        # Calculate and return cosine similarity
        return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)
