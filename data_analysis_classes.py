import scipy as sp
import sympy as sy
import sympy.utilities.lambdify as lambdify
import Assorted_Statistics as ass

class model_construct:

    def __init__(self, model, parameters, indep_var):
        """
        
        :param model: sympy model
        :param parameters: List of sy.Symbol
        :param indep_var:
        """
        self.model = model
        self.parameters = parameters
        self.indep_var = indep_var

        self.model_grad = sy.diff(model, *parameters)

    def model_num(self, args):
        """
        
        :param args: 
        :return: 
        """
        func = lambdify([self.parameters, self.indep_var], self.model)
        return func(*args)

    def model_grad_num(self, args):
        """
        
        :param args: 
        :return: 
        """
        func = lambdify([self.parameters, self.indep_var], self.model_grad)
        return func(*args)
