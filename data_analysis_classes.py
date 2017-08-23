import scipy as sp
from scipy import linalg as la
import sympy as sy
import sympy.utilities.lambdify as lambdify
import Assorted_Statistics as ass

class model_construct:

    def __init__(self, model, parameters, indep_var):
        """
        
        :param model: The model given in terms of sympy functions and symbols.
        :param parameters: List of sy.Symbols representing the model parameters.
        :param indep_var: List of sy.Symbols representing the model independet variables.
        """
        self.model = model
        self.parameters = parameters
        self.indep_var = indep_var

        self.model_grad = [sy.diff(model, i) for i in parameters]
        self.model_hessian = [[sy.diff(model, i) * sy.diff(model, j) for j in parameters] for i in parameters]

    def model_num(self, args):
        """
        Evaluate the model for numeric values of the parameters and independent variables.
        :param args: A list structured as [[parameters], [indep_var]].
        :return: Numerical evaluation of the model.
        """
        func = lambdify([self.parameters, self.indep_var], self.model)
        return func(*args)

    def model_grad_num(self, args):
        """
        Evaluate the model gradient for numeric values of the parameters and independent variables.
        :param args: A list structured as [[parameters], [indep_var]].
        :return: Numerical evaluation of the model gradient.
        """
        func = lambdify([self.parameters, self.indep_var], self.model_grad)
        return sp.asarray(func(*args))

    def model_hessian_num(self, args):
        """
        Evaluate the model Hessian for numeric values of the parameters and independent variables.
        Technically this is not the actual Hessian, but the one that is typically used for nonlinear fits.
        This is explained in William Press, Numerical Recipies (Third Edition), section 15.5.1 page 801.
        :param args: A list structured as [[parameters], [indep_var]].
        :return: Numerical evaluation of the model Hessian.
        """
        func = lambdify([self.parameters, self.indep_var], self.model_hessian)
        return func(*args)

class levenberg_marquardt:

    def __init__(self, a_model_obj, parameter_guess, the_data, indep, sigs):
        """
        Initialize object for optimizing the parameters of a model via the Levenberg-Marquardt method.
        :param a_model_obj: An instance of model_construct representing the model to be fit.
        :param parameter_guess: An array containing the initial guess for the parameters.
        :param the_data: An array containing the measured date.
        :param sigs: the measurement uncertainty for each measured value.
        """
        self.a_model_obj = a_model_obj
        self.parameter_guess = parameter_guess
        self.the_data = the_data
        self.indep = indep
        self.sigs = sigs

        self.current_parameters = parameter_guess

        self.theory = sp.asarray([self.a_model_obj.model_num([[self.current_parameters], [i]]) for i in self.indep])
        self.cs = ass.chi_squared(self.the_data, self.theory, self.sigs)

        self.n = len(the_data)
        self.cs_error = 0.001 * self.n
        self.occurences = 5
        self.max_it = 50
        self.current_it = [0]

        self.lam = 0.001

        self.beta = -sum([self.a_model_obj.model_grad_num([[self.current_parameters], [h]]) * (i - j) / k**2 for
                         h, i, j, k in zip(self.indep, self.the_data, self.theory, self.sigs)]) / 2
        self.alpha = -sum([self.a_model_obj.model_hessian_num([[self.current_parameters], [i]]) / j**2 for i, j in
                           zip(self.indep, self.sigs)]) / 2
        self.alpha_prime = self.alpha + self.lam * sp.diag(sp.diag(self.alpha))

    def func_eval(self, func, param):
        """
        Evaluate a function for every measured data point.
        :param func: The function to be evaluated.
        :param param: The parameters to evaluate the function with.
        :return: An array containing the function evaluated for each measured value.
        """
        evaluated = sp.zeros(len(self.indep))
        for i, j in zip(self.indep, evaluated):
            j = func([[param], [i]])

        return evaluated

    def lev_mar_update(self, test_cs, test_param, d_chi, d_chi_checks, theory):
        """
        Update self.theory (and all the attributes that depend on it) according to the Levenberg-Marquardt method.
        :param test_cs: The test update for self.cs.
        :param test_param: The test parameters.
        :return: None.
        """
        if test_cs <= self.cs:
            self.current_parameters = test_param
            self.lam = 0.1 * self.lam
            self.cs = test_cs
            self.theory = theory
            d_chi_checks.append(abs(d_chi))
            print("fuckit")
        else:
            self.lam = 10 * self.lam

        return None

    def lev_mar_step(self):
        """
        Update the current_parameter (not the attribute) via the Levenberg-Marquardt method.
        :return: The updated parameters.
        """
        alpha_prime = self.alpha + self.lam * sp.diag(sp.diag(self.alpha))
        a = la.inv(alpha_prime).dot(self.beta)
        return self.current_parameters + a

    def check_stop(self):
        """
        Prevent the algorithm from never ending.
        :return: None.
        """
        self.current_it[0] = self.current_it[0] + 1
        if self.current_it[0] > self.max_it:
            raise Exception("TOO MANY ITERATIONS!!!")

        return None

    def lev_mar_run(self):
        """
        Carry out the Levenberg-Marquardt method to optimize the parameters for the given model.
        :return: None.
        """
        checks = []
        while len(checks) < self.occurences:
            param_new = self.lev_mar_step()
            theory_new = sp.asarray([self.a_model_obj.model_num([[param_new], [i]]) for i in self.indep])
            cs_new = ass.chi_squared(self.the_data, theory_new, self.sigs)

            print(self.current_parameters, param_new, self.cs, cs_new)

            d_cs = cs_new - self.cs
            self.lev_mar_update(cs_new, param_new, d_cs, checks,  theory_new)

        while not all(sp.asarray(checks[-self.occurences:]) < self.cs_error):
            print('fuck')
            param_new = self.lev_mar_step()
            theory_new = sp.asarray([self.a_model_obj.model_num([[param_new], [i]]) for i in self.indep])
            cs_new = ass.chi_squared(self.the_data, theory_new, self.sigs)

            d_cs = cs_new - self.cs

            self.lev_mar_update(cs_new, param_new, d_cs, checks, theory_new)
            self.check_stop()

        return None
