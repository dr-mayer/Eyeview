/**
 * Created by Nathan on 4/11/16.
 */
import java.util.Arrays;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;


public class WinRateFit {

    //This is the data
    final static double bids[] = new double[] {
            662,  1959,  5054,  7344,  8287,  6289,  3847,  2364,
            1584,  1634,   979,   800,   613,   264,    67,    54,
            115,    74,   153,  6564, 42482,  6711,  5405,  1175,
            9128,  2316};

    final static double bid_price[] = new double[] {
            3.6405,  4.5853,  5.5409,  6.5296,  7.5049,  8.484,
            9.4098, 10.4314, 11.4872, 12.4869, 13.4687, 14.4698,
            15.4666, 16.4889, 17.32,   18.2865, 19.2853, 20.5238,
            24.0228, 25.0051, 30,      35.0002, 40.0003, 44.9937,
            50,      55};

    final static double wins[] = new double[] {
            13,   76,  292,  658,  982,  795,  501,  373,  280,  291,
            195,  164,  129,   54,   15,   11,   28,   12,   23, 1724,
            10579, 1972, 1384,  341, 2326,  671};

    //This is logistic function used for the Win-Rate Curve
    public static double par3_logistic(double x, double par[]) {
        double par0 = par[0];
        double par1 = par[1];
        double par2 = par[2];
        double ex    = -par0*(x - par1);
        double lt = par2/(1 + Math.exp(ex));
        return(lt);
    }

    //This calculates the log-likelihood metric for ONE data point.  The total log-likelihood is
    //the sum of all the data points.  That summation is handeled below.
    public static double log_likelihood(double price, double n_bid, double n_win, double par[]) {
        double model  = par3_logistic(price, par) * n_bid;
        //System.out.println("price: " + price + " bids: " + n_bid + " wins: " + n_win + " model: " + model);

        //The defination of log-likelihood is the term that contains the logarithm is ZERO when the observed counts
        // (n_win) is ZERO.
        double l_term = (n_win == 0) ? 0 : -n_win + n_win*( Math.log(n_win) - Math.log(model) );
        return(2 * (model + l_term ));
    }

    public static void main(String[] args) {

        SimplexOptimizer optimizer = new SimplexOptimizer(1e-10, 1e-30);  /*These parameters have to do with the allowed
        tolerance for the parameters and the log-likelihood near the best fit. */
        final LogLikelihood logLikelihood = new LogLikelihood();

        final PointValuePair optimum =
                optimizer.optimize(
                        new MaxEval(1000),
                        new ObjectiveFunction(logLikelihood),
                        GoalType.MINIMIZE,
                        new InitialGuess(new double[]{ 0.5, 10, 0.5 }),   //This is the starting fit parameters to try
                        new NelderMeadSimplex(new double[]{ 1, 1, 0.3})); /*This is the size of the "Simplex" in
                        n-dimensions. Bigger is better up to a point.  If the Simplex is to small, then the algo
                        may converge on a local instead of global mimimum.  If parameter c is big however The
                        log-likelihood will through an error (the logarithm of a negative number)*/

        double par[] = optimum.getPoint();

        /*
        double a = optimum.getPoint().getEntry(0);
        double b = optimum.getPoint().getEntry(1);
        double c = optimum.getPoint().getEntry(2);
        */

        System.out.println("");
        System.out.println("par[0]: " + par[0] + " par[1]: " + par[1] + " par[2]: " + par[2]);
        System.out.println("Log-Likelihood: " + optimum.getSecond());


    }

    //This
    private static class LogLikelihood implements MultivariateFunction {

        public double value(double[] par) {

            double ll = 0;
            for(int i = 0; i < bid_price.length; i++) ll = ll + log_likelihood(bid_price[i], bids[i], wins[i], par);
            //System.out.println("price: " + bid_price[i] + " bids: " + bids[i] + " wins: " + wins[i] + " LL: " + ll);


            //Print out the fit result at this iteration.
            System.out.println("par[0]: " + par[0] + " par[1]: " + par[1] + " par[2]: " + par[2] + " Loglikelihood: " + ll);
            return(ll);
        }
    }
}