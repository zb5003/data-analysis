import scipy as sp
from scipy import linalg as la
import sympy as sy
import random
import sympy.utilities.lambdify as lambdify
import matplotlib.pyplot as plt
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
        :param indep: Float array. Contains the values of the independent variables.
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

            d_cs = cs_new - self.cs
            self.lev_mar_update(cs_new, param_new, d_cs, checks,  theory_new)

        while not all(sp.asarray(checks[-self.occurences:]) < self.cs_error):
            param_new = self.lev_mar_step()
            theory_new = sp.asarray([self.a_model_obj.model_num([[param_new], [i]]) for i in self.indep])
            cs_new = ass.chi_squared(self.the_data, theory_new, self.sigs)

            d_cs = cs_new - self.cs

            self.lev_mar_update(cs_new, param_new, d_cs, checks, theory_new)
            self.check_stop()

        return None

class monte_carlo:

    def __init__(self, indep, y, sig, model_func, param_values):
        """
        :param indep: Float array. Contains numeric values of the independent variable.
        :param y: Float array. Measured values (floats).
        :param sig: Float array. Measurement uncertainties (floats).
        :param model_func: model_construct object. Symbolic function to be fitted.
        :param param_values: Initial guess for the values of the model parameters.
        """
        self.indep = indep
        self.y = y
        self.sig = sig  # measurement error
        self.model_func = model_func
        self.param_values = param_values  # values of model parameters (numbers)

        self.param_len = len(model_func.parameters)
        self.meas_len = len(y)

    def generate_measurement_sets(self):
        """
        Generate a simulated set of measurements with their corresponding independent variables and measurement errors.
        :return: List of ndarrays. The first array are the measurements, the second is the independent variables
        (this array might have more than one row), and the third is an array of the measurement errors.
        """
        return NotImplementedError

    def generate_simulated_parameters(self):
        """
        Run the Monte Carlo algorihtm.
        :return: Float array. Each row is an individual fit and the columns are the resulting model parameter values.
        """
        return NotImplementedError

    def parameter_average(self, sim):
        """
        Calculate the average value of each model parameter over the distribution of all Monte Carlo simulations.
        :param sim: Ndarray.  Contains the model parameter values calculated for each Monte Carlo simulation.
        :return: Ndarray. Contains the average values of the model parameters.
        """
        averages = sp.zeros(self.param_len)
        for param in range(self.param_len):
            averages[param] = ass.average(sim[:, param])
        return averages

    def parameter_stddev(self, sim):
        """
        Calculate the standard deviation for each model parameter over the distribution of all Monte Carlo simulations.
        :param sim: Ndarray. Contains the model parameter values calculated for each Monte Carlo simulation.
        :return: Ndarray. Contains the standard deviations of the model parameters.
        """
        stddev = sp.zeros(self.param_len)
        for param in range(self.param_len):
            stddev[param] = ass.stddev(sim[:, param])
        return stddev

    def run_monte_carlo(self):
        """
        Run the Monte Carlo algorithm and calculate statistics on the model parameters.
        :return:
        """
        return NotImplementedError

class bootstrap(monte_carlo):

    def __init__(self, indep, y, sig, model_func, param_values, n_sample_sets=100):
        monte_carlo.__init__(self, indep, y, sig, model_func, param_values)
        self.n_sample_sets = n_sample_sets

    def generate_measurement_sets(self):
        """
        Generate a simulated data set by replacement.
        :return: 3 1-D ndarrays. Contains simulated measured values, simulated measurement
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

    def generate_simulated_parameters(self):
        """
        Use the Levenberg-Marquardt method to fit a number of simulated data sets to a model.
        :return: Ndarray containing the fitted parameters for a number of simulated data sets.
        """
        sim = sp.zeros((self.n_sample_sets, self.param_len))
        for i in range(self.n_sample_sets):
            measured, sig, indep = self.generate_measurement_sets()
            single_sim = levenberg_marquardt(self.model_func, self.param_values, measured, indep, sig)
            single_sim.lev_mar_run()
            sim[i, :] = single_sim.current_parameters
            print("Generated measurement set", i, "fits with parameters", sim[i, :])
        return sim

    def run_monte_carlo(self):
        """
        Run the Monte Carlo algorithm and calculate statistics on the model parameters.
        :return: Tuple.  The first element is an ndarray containing the average parameters,
                         the second is an ndarray containing the stddev on the parameters,
                         and the third is an ndarray containing the parameter results for each simulation.
        """
        simulated_parameters = self.generate_simulated_parameters()
        parameter_average_value = self.parameter_average(simulated_parameters)
        parameter_stddev_value = self.parameter_stddev(simulated_parameters)
        return parameter_average_value, parameter_stddev_value, simulated_parameters

    def save_data(self, folder=''):
        """

        :param folder:
        :return:
        """
        aves, stds, all_params = self.run_monte_carlo()
        sp.savetxt(folder + "parameter_data.txt", all_params)
        n = len(aves)
        fig, ax = plt.subplots(nrows=n, ncols=1)
        fig.subplots_adjust(hspace=0.3 * n)
        fig.suptitle(r"Parameter Distributions for model $" + str(sy.latex(self.model_func.model)) + "$")
        for i in range(n):
            ax[i].hist(all_params[:, i])
            ax[i].set_title("Distribution of Parameter " + str(self.model_func.parameters[i]))
            ax[i].set_xlabel("Value of Parameter")
            ax[i].set_ylabel("Bin Occupation")
            ax[i].axvline(aves[i], linewidth='2', color='red')
            ylim = ax[i].get_ylim()
            ax[i].errorbar(aves[i], (ylim[1] - ylim[0]) / 2, xerr=stds[i], linestyle='none', elinewidth=2, ecolor='red', capthick=2)
        plt.savefig(folder + 'parameters.png')
        plt.close()

        return aves, stds, all_params