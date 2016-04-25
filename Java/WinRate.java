/**
 * Created by Nathan on 4/20/16.
 */

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;

public class WinRate {

    //Data used to fit the win rate curve
    protected double[] data_bids;
    protected double[] data_wins;
    protected double[] data_bidPrice;
    protected double   floor_price;

    //Results of the curve
    protected double[] bestFit;
    protected double[] initialGuess;
    protected double[] simplexConfig;
    protected double logLikelihood;
    protected int nDataPoints;

    //Constructors
    /***********************************************/
    //Standard Constructor
    public WinRate(double[] bids, double[] wins,
                   double[] totalBp, double[] guess,
                   double fl_price) {
        //Initialize variables
        data_bidPrice = totalBp;
        for(int i=0; i<totalBp.length; i++) data_bidPrice[i] = data_bidPrice[i]/bids[i];

        data_bids     = bids;
        data_wins     = wins;
        initialGuess  = guess;
        simplexConfig = new double[]{ 1, 1, 0.3};
        floor_price   = fl_price;
        nDataPoints   = data_bidPrice.length;

        Fit();
    }

    //Constructor with default initial guess
    public WinRate(double[] bids, double[] wins,
                   double[] totalBp, double fl_price ) {
        //Initialize variables
        data_bidPrice = totalBp;
        for(int i=0; i<totalBp.length; i++) data_bidPrice[i] = data_bidPrice[i]/bids[i];

        data_bids     = bids;
        data_wins     = wins;
        initialGuess  = new double[]{ 0.5, 7, 0.2 };
        simplexConfig = new double[]{ 1, 1, 0.1};
        floor_price   = fl_price;
        nDataPoints   = data_bidPrice.length;

        Fit();  //Now run the fit;
    }
    /***********************************************/
    //End Constructors


    //Setter method
    public void InitialGuess(double[] guess) {
        initialGuess = guess;
    }

    //Getter methods
    /***********************************************/
    public double[] GetFitParameters() {
        return bestFit;
    }

    public double[] BestFit() {
        return bestFit;
    }

    public double GetLogLikelihood() {
        return logLikelihood;
    }

    public double ReducedLogLikelihood() {
        return logLikelihood/(nDataPoints - bestFit.length);
    }
    /***********************************************/
    //End Getter methods


    //Utility methods
    /***********************************************/

    //The Model that is actually fit.
    protected double Model(double bid_price, double[] par_array) {
        //The model is a Fermi or Logistic Function.
        double par0 = par_array[0];
        double par1 = par_array[1];
        double par2 = par_array[2];
        double ex   = -par0*(bid_price - par1);
        return  par2/(1 + Math.exp(ex));
    }

    //The model for after a fit has been performed
    public double Predict(double bid_price) {
        return Model(bid_price, bestFit);
    }

    //Run the fit.
    protected void Fit() {
        SimplexOptimizer optimizer = new SimplexOptimizer(1e-10, 1e-30);  /*These parameters have to do with the allowed
        tolerance for the parameters and the log-likelihood near the best fit. */
        final LogLikelihood ll = new LogLikelihood();

        final PointValuePair optimum =
                optimizer.optimize(
                        new MaxEval(1000),
                        new ObjectiveFunction(ll),
                        GoalType.MINIMIZE,
                        new InitialGuess(initialGuess),   //This is the starting fit parameters to try
                        new NelderMeadSimplex(simplexConfig) ); /*This is the size of the "Simplex" in
                        n-dimensions. Bigger is better up to a point.  If the Simplex is to small, then the algo
                        may converge on a local instead of global mimimum.  If parameter c is big however The
                        log-likelihood will throw an error (the logarithm of a negative number)*/

        bestFit       = optimum.getPoint();
        logLikelihood = optimum.getSecond();

    }
    /***********************************************/
    //End Utility methods.

    //This is the goodness of fit metric.
    protected class LogLikelihood implements MultivariateFunction {

        public double value(double[] par_array) {
            double ll = 0;

            for (int i = 0; i < data_bids.length; i++) {
                double model = Model(data_bidPrice[i], par_array) * data_bids[i];
                double l_term = (data_wins[i] == 0) ? 0 : -data_wins[i] + data_wins[i] * (Math.log(data_wins[i]) - Math.log(model));
                System.out.println("Win Rate:" + ll);
                ll = ll + 2 * (model + l_term);
            }
            return ll;
        }
    }
}
