
/**
 * Created by Nathan on 4/21/16.
 */

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;


public class WinPrice {


    //Data to use in the fit.
    protected double[] data_wins;
    protected double[] data_bidPrice;
    protected double[] data_winPrice;
    protected double[] data_winPriceSqr;
    protected double[] winPriceErr;

    //Fit configuration variables
    protected double[] initialGuess;
    protected double[] simplexConfig;
    protected double floor_price;

    //Results
    protected double[] bestFit;
    protected double chi2;
    protected int nDataPoints;

    //Constructor method
    /***********************************************/
    public WinPrice(double[] wins, double[] bids,
                    double[] totalBp, double[] totalWp,
                    double[] totalWpSqr, double fl_price) {

        data_bidPrice    = totalBp;
        data_winPrice    = totalWp;
        data_winPriceSqr = totalWpSqr;

        data_wins        = wins;
        initialGuess     = new double[]{4, 5, 10};
        simplexConfig    = new double[]{2, 2, 2};
        floor_price      = fl_price;
        winPriceErr      = new double[totalWpSqr.length];
        nDataPoints      = 0;

        //Fill in the uncertainty in the win_price;
        for(int i = 0; i < data_winPriceSqr.length; i++) {
            data_bidPrice[i] = data_bidPrice[i]/bids[i];
            if(wins[i] != 0 ) {
                data_winPrice[i]    = data_winPrice[i] / wins[i];
                data_winPriceSqr[i] = data_winPriceSqr[i] / wins[i];
                winPriceErr[i]      = data_winPriceSqr[i] - data_winPrice[i] * data_winPrice[i];
            }
            else {
                winPriceErr[i]      = 0;
                data_winPrice[i]    = 0;
                data_winPriceSqr[i] = 0;
            }
            nDataPoints++;
        }

        Fit();

    }

    public WinPrice(double[] wins, double[] bids,
                    double[] totalBp, double[] totalWp,
                    double[] totalWpSqr, double[] guess,
                    double fl_price) {
        data_bidPrice    = totalBp;
        data_winPrice    = totalWp;
        data_winPriceSqr = totalWpSqr;

        data_wins        = wins;
        initialGuess     = guess;
        simplexConfig    = new double[]{2, 2, 2};
        floor_price      = fl_price;
        winPriceErr      = new double[totalWpSqr.length];
        nDataPoints      = 0;

        for(int i = 0; i < data_winPriceSqr.length; i++) {
            data_bidPrice[i] = data_bidPrice[i]/bids[i];
            if(wins[i] != 0 ) {
                data_winPrice[i]    = data_winPrice[i] / wins[i];
                data_winPriceSqr[i] = data_winPriceSqr[i] / wins[i];
                winPriceErr[i]      = data_winPriceSqr[i] - data_winPrice[i] * data_winPrice[i];
            }
            else {
                winPriceErr[i]      = 0;
                data_winPrice[i]    = 0;
                data_winPriceSqr[i] = 0;
            }
            nDataPoints++;
        }

        Fit();

    }
    /***********************************************/
    //End Constructor methods

    //Setter methods
    /***********************************************/
    public void SetInitialGuess(double[] guess) {
        initialGuess = guess;
    }

    /***********************************************/
    //End setter methods

    //Getter methods
    /***********************************************/
    public double[] BestFit() {
        return bestFit;
    }

    public double GetChi2() {
        return chi2;
    }

    public double ReducedChi2() {
        return chi2/(nDataPoints - bestFit.length);
    }
    /***********************************************/
    //End Getter methods

    //Utility methods
    /***********************************************/

    //This is the log-logistic or log-fermi function used for the Win-Rate Curve
    //It is the actuall mathematical model that we are attempting to fit.
    protected double Model(double bid_price, double[] par_array) {
        double par0 = par_array[0];
        double par1 = par_array[1];
        double par2 = par_array[2];
        double ex   = -par0*(bid_price - par1);
        return par2*Math.log(bid_price+1)/(1 + Math.exp(ex));
    }

    //The model prediction
    public double Predict(double bid_price) {
        return Model(bid_price, bestFit);
    }

    //Run the fit.
    protected void Fit() {
        SimplexOptimizer optimizer = new SimplexOptimizer(1e-10, 1e-30);  /*These parameters have to do with the allowed
        tolerance for the parameters and the log-likelihood near the best fit. */
        final Chi2 chi_squared = new Chi2();

        final PointValuePair optimum =
                optimizer.optimize(
                        new MaxEval(1000),
                        new ObjectiveFunction(chi_squared),
                        GoalType.MINIMIZE,
                        new InitialGuess(initialGuess),   //This is the starting fit parameters to try
                        new NelderMeadSimplex(simplexConfig) ); /*This is the size of the "Simplex" in
                        n-dimensions. Bigger is better up to a point.  If the Simplex is to small, then the algo
                        may converge on a local instead of global mimimum.  If parameter c is big however The
                        log-likelihood will throw an error (the logarithm of a negative number)*/

        bestFit       = optimum.getPoint();
        chi2          = optimum.getSecond();

    }
    /***********************************************/
    //End Utility functions

    //This is the goodness of fit metric.
    protected class Chi2 implements MultivariateFunction {

        public double value(double[] par_array) {
            double c2 = 0;

            for (int i = 0; i < data_winPrice.length; i++) {
                if(data_wins[i] != 0 && data_bidPrice[i] > 1) {
                    double model = Model(data_bidPrice[i], par_array);
                    double term = (data_winPrice[i] - model) * (data_winPrice[i] - model) / winPriceErr[i];
                    System.out.println("Win Price:" + c2);
                    c2 = c2 + term;
                }
            }
            return c2;
        }
    }
}
