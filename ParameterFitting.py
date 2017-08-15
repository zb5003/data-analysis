import sympy.utilities.lambdify as lambdify
import scipy as sp
from scipy import linalg as la
import scipy.special as spsp
import random
import time

class Levenberg_Marquardt:
    """Perform the Levenberg-Marquardt nonlinear least squares fitting method.
    The Purpose is to analyze a particular 1-D model.
    References: [1] William Press, Numerical Recipes 1987."""

    end_count = 4  # Number of times in a row chi must have a small change to end the loop.
    error_chi = 10  # Target value of chi squared.
    error_grad = 1

    def __init__(self, model, indep, model_param):
        """
        Basically everything except levmar is sympy in sympy out.
        ORDER OF THE INDEP AND NODEL_PARAM MATTERS!!!
        :param model: Symbolic function to be fitted.
        :param indep: Symbolic independent variable
        :param model_param: A list containing the sympy variables that model depends on.
        """
        self.model = model
        self.indep = indep
        self.length = len(model_param)  # Number of model parameters
        self.model_param = model_param  # sympy objects

    def chisquared(self, y, sig, x):
        """
        Find the symbolic representation of chi squared in terms of the model parameters.
        :param y:  Array of measured values (floats).
        :param sig: Array of measurement uncertainties (floats).
        :param x: Array containing numeric values of the independent variable.
        :return: The symbolic chi squared sum for a particular model.
        """
        chisq = 0
        for i in range(len(y)):
            chisq = chisq + ((y[i] - self.model.subs(self.indep, x[i])) / sig[i])**2
        return chisq

    def chisquared_num(self, y, sig, x, param):
        """
        Calculate the numerical value of chi squared for a model given parameter values.
        :param y: Array of measured values (floats).
        :param sig: Array of measurement uncertainties (floats).
        :param x: Array containing numeric values of the independent variable.
        :param param: List containing numeric values of the model parameters.
        :return: Numerical value of chi squared.
        """
        chi = self.chisquared(y, sig, x)
        chi_func = lambdify(self.model_param, chi)
        return chi_func(*param)

    def chisq_grad(self, y, sig, x):
        """
        Returns a symbolic expression for the gradient of chi squared in terms of the model parameters.
        :param y:  Array of measured values (floats).
        :param sig: Array of measurement uncertainties (floats).
        :param x: Array containing numeric values of the independent variable.
        :return: A list containing the symbolic gradient of chi squared.
        """
        c = []
        for i in range(self.length):
            c.append(self.chisquared(y, sig, x).diff(self.model_param[i]))
        return c

    def beta(self, y, sig, x, param):
        """
        Calculate half the negative gradient of chi squared.
        Might be on the chopping block.
        :param y: Array of measured values (floats).
        :param sig: Array of measurement uncertainties (floats).
        :param x: Array containing numeric values of the independent variable.
        :param param: List containing numeric values of the model parameters.
        :return: Array containing the negative of half the numeric gradient.
        """
        c = self.chisq_grad(y, sig, x)
        d = lambdify(self.model_param, c)
        return -1 / 2 * sp.asarray(d(*param))

    def chisq_hessian(self, y, sig, x):
        """
        Find the symbolic form of the Hessian of chi squared, assuming random measurement error, in terms of the model
        parameters.
        :param y:  Array of measured values (floats).
        :param sig: Array of measurement uncertainties (floats).
        :param x: Array containing numeric values of the independent variable.
        :return: NxN list containing the symbolic Hessian for chi squared (N is the number of model parameters).
        """
        hes = [[] for ii in range(self.length)]  # A list of length self.length containing empty lists.
        for i in range(self.length):
            for j in range(self.length):
                # two_alpha = 0
                # for k in range(len(y)):
                #     two_alpha = self.
                temp = self.chisquared(y, sig, x).diff(self.model_param[i]).diff(self.model_param[j])
                hes[i].append(temp)
        return hes

    def alpha(self, y, sig, x, param):
        """
        Returns a numeric expression for the Hessian matrix of chi squared.
        Note: The second derivatives are excluded and this expression only contains the product of first derivatives.
              See Numerical Recipes page 800-801 (Press, 1987) for more details.
        :param y:  Array of measured values (floats).
        :param sig: Array of measurement uncertainties (floats).
        :param x: Array containing numeric values of the independent variable.
        :param param: List containing numeric values of the model parameters.
        :return: Array containing half of the numeric hessian.
        """
        c = self.chisq_hessian(y, sig, x)
        d = lambdify(self.model_param, c)
        return 1 / 2 * sp.asarray(d(*param))

    def levmar_step(self, y, sig, x, param, lambd):
        """
        Carry out one step of the Levenberg-Marquart algorithm.
        :param y:  Array of measured values (floats).
        :param sig: Array of measurement uncertainties (floats).
        :param x: Array containing numeric values of the independent variable.
        :param param: List containing numeric values of the model parameters.
        :param lambd: Levenberg-Marquart fudge factor (float).
        :return: Array containing the numeric values of the model parameters after one step of the Levenberg-Marquart
                 algorithm.
        """
        alpha = self.alpha(y, sig, x, param)
        m = alpha + sp.multiply(lambd * sp.identity(self.length), alpha)
        m_inv = la.inv(m)
        beta = self.beta(y, sig, x, param)
        return m_inv.dot(beta)

    def levmar(self, y, sig, x, param):
        """
        Carry out the Levenberg-Marquardt least squares algorithm.
        [1] pg. 801-806.
        :param y: Array of measured values (floats).
        :param sig: Array of measurement uncertainties (floats).
        :param x: Array containing numeric values of the independent variable.
        :param param: List containing numeric values of the model parameters.
        :return: Array with parameters that minimize chi squared.
        """
        t1 = time.time()
        tracker = 0  # Number of iterations of the loop.
        counter = 0  # Number of iterations in a row chi squared has been below threshold.
        chi_i = self.chisquared_num(y, sig, x, param)  # initial chis squared value
        lambd = 0.001

        while counter < Levenberg_Marquardt.end_count:
            dparam = self.levmar_step(y, sig, x, param, lambd)  # Amount to change the parameters by.
            chi_f = self.chisquared_num(y, sig, x, param + dparam)  # Updated chi squared value.
            tracker = tracker + 1

            if abs(chi_i - chi_f) < Levenberg_Marquardt.error_chi:
                counter = counter + 1
            else:
                counter = 0

            if chi_f < chi_i:
                lambd = lambd * 0.1
                param = param + dparam
                chi_i = chi_f
            else:
                lambd = lambd * 10

        t2 = time.time()
        print(str(tracker) + " iterations took " + str(t2 - t1) + " second.")
        return param

    def Q(self, nu, chisq):
        """
        Calculate the probability that a normally distributed random numbers produce a chi squared value greater than
        measured. The distribution is described by an incomplete gamma function and should only be used for models that
        are linear or close to linear in their parameters.
        [1] pg. 778-780.
        :param nu: Number of degrees of freedom.
        :param chisq: Measured chi squared value.
        :return: The probability that random noise will produce a chi squared greater than the measured value.
        """
        return 1 - spsp.gammainc(nu / 2, chisq / 2)

class Monte_Carlo:
    """Various Monte Carlo algorithms to estimate parameter uncertainty.
    The resampled values are the terms in the chi squared sum, (ymeasured - ytheory)/sig."""

    samples = 100

    def __init__(self, indep, indep_var, y, sig, model, model_param, param_values):
        """

        :param indep: Array containing numeric values of the independent variable.
        :param indep_var: Symbolic independent variable
        :param y: Array of measured values (floats).
        :param sig: Array of measurement uncertainties (floats).
        :param model: Symbolic function to be fitted.
        :param model_param: A list containing the sympy variables that model depends on.
        :param param_values: Initial guess for the values of the model parameters.
        """
        self.indep = indep
        self.indep_var = indep_var
        self.y = y
        self.sig = sig  # measurement error
        self.model = model
        self.model_param = model_param
        self.param_values = param_values  # values of model parameters (numbers)
        self.param_len = len(model_param)
        self.meas_len = len(y)

    def sim_meas(self):
        """
        Generate a simulated data set by replacement.
        :return: 3 1-D Arrays (floats): simulated measured values, simulated measurement
                 uncertainties, and simulated independent variables.
        """
        n_replace = int(random.random() * self.meas_len / 4 - 1)  # number of measured values to replace/duplicate
        sim_measured = self.y.copy()
        sim_sig = self.sig.copy()
        sim_indep = self.indep.copy()
        for i in range(n_replace):
            n = int((self.meas_len - 1) * random.random())
            sim_measured[n] = self.y[n + 1]
            sim_sig[n] = self.sig[n + 1]
            sim_indep[n] = self.indep[n + 1]
        return sim_measured, sim_sig, sim_indep

    def mc(self):
        """
        Use the Levenberg-Marquardt method to fit a number of simulated data sets to a model.
        :return: Array containing the fitted parameters for a number of simulated data sets.
        """
        sim = sp.zeros((Monte_Carlo.samples, self.param_len))
        sim_single = Levenberg_Marquardt(self.model, self.indep_var, self.model_param)
        for i in range(Monte_Carlo.samples):
            measured, sig, indep = self.sim_meas()
            a = sim_single.levmar(measured, sig, indep, self.param_values)
            sim[i, :] = a
            print(sim[i, :], i)
        return sim

    def ave(self, sim):
        """Compute the average of each parameter."""
        averages = sp.zeros(self.param_len)
        for i in range(self.param_len):
            averages[i] = sum(sim[:, i]) / Monte_Carlo.samples
        return averages

    def stddev(self, sim, averages):
        """Compute the standard deviation of each parameter."""
        standard_dev = sp.zeros(self.param_len)
        for i in range(self.param_len):
            standard_dev[i] = sum(sim[:, i]**2) / Monte_Carlo.samples - averages[i]**2
        return standard_dev

    def stat(self):
        """
        Calculate various statistics for the simulated data.
        :return: The simulated data, averaged simulation parameters, uncertainty on the simulated parameters,
                 chi squared for the simulated parameters, and the Q for the simulated chi squared.
        """
        sim = self.mc()
        average = self.ave(sim)
        std = self.stddev(sim, average)

        stat_sim = Levenberg_Marquardt(self.model, self.indep_var, self.model_param)
        chisq = stat_sim.chisquared_num(self.y, self.sig, self.indep, average)
        Q = stat_sim.Q(self.meas_len - self.param_len, chisq)

        return sim, average, std, chisq, Q
