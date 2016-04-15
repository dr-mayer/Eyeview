/**
 * Created by Nathan on 4/7/16.
 */
import org.apache.commons.math3.fitting.leastsquares.*;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.util.Pair;




public class WinPriceFit {

    //This is the log-logistic function used for the Win-Rate Curve
    //It is the actuall mathematical model that we are attempting to fit.
    public static double par3_llt(double bid_price, double[] par) {
        double par0 = par[0];
        double par1 = par[1];
        double par2 = par[2];
        double ex    = -par0*(bid_price - par1);
        double llt = par2*Math.log(bid_price)/(1 + Math.exp(ex));
        return(llt);
    }

    public static void main(String[] args) {

        //These 5 doubles are the data used for calculating the win-cost curve.

        //The average of the win price squared eg. average(win_price*win_price)
        //By bid price bucket
        final double win_price_avgsqr[] = new double[] {
                211.2732,  20.0162, 101.3665,  59.5519, 103.5344, 149.8282, 207.9738, 279.8721, 44.2031, 7.3617,
                 12.3523,  33.5181, 264.4375,  27.0668, 119.3962,  82.5744,  49.7778,  38.7091, 228.558,  71.0065};

        //The average win price
        //By bid price bucket
        final double win_price_avg[] = new double[] {
            12.432, 4.3343, 9.1902, 7.1826, 9.3597, 10.8438, 12.4439, 13.8145, 6.3385, 2.6492, 3.4145, 5.5701, 13.8111,
                    5.0203, 9.8783, 8.4115, 6.6496, 5.9582, 12.967, 7.7523};

        //The average bid price
        //By bid price bucket
        final double bid_price_avg[] = new double[] {
            35.0005, 5.5409, 14.4698, 11.4872, 16.3521, 25.0245, 30.0024, 51.0119, 9.4098, 3.6405, 4.5855, 7.5049, 45,
                     6.5296, 22.852,  13.4687, 10.4314,  8.484,  40.0063, 12.4869};

        //The number of wins
        //By bid price bucket
        final double n_wins[] = new double [] {
                1973,  982, 10589, 291,  76, 501, 1385, 164,  658,
                 261,   52,    13, 340, 292, 280,  373, 195, 1734,
                795,  2997 };



        //This is the object necessary to plug into the least squares builder for the fitting algorithm
        MultivariateJacobianFunction log_Logistic = new MultivariateJacobianFunction() {
            public Pair<RealVector, RealMatrix> value(final RealVector point) {

                //These are the free parameters for the fit.
                //Recomend that this get implemented in such
                //a way that additional parameters can be
                //easily added.
                double[] par = new double[] {point.getEntry(0), point.getEntry(1), point.getEntry(2)};

                //The RealVector stores the values for the model at THIS iteration of the fit.
                //One entry for every fit point there is.
                RealVector value = new ArrayRealVector(bid_price_avg.length);
                //The RealMatrix stores the values for the derivative at THIS iteration of the fit
                //One row for every fit point there is
                //The number of columns MUST be the same as the number of paramters (a,b,c) in this case three3.
                RealMatrix jacobian = new Array2DRowRealMatrix(bid_price_avg.length, 3);

                for (int i = 0; i < bid_price_avg.length; ++i) {
                    double bid_price = bid_price_avg[i];
                    double f = par3_llt(bid_price, par); //Values of the model

                    value.setEntry(i,  f);
                    //derivatives refactored wrt to the original function "f"
                    // derivative with respect to a
                    jacobian.setEntry(i, 0,
                            (Math.log(bid_price)*f*f*(par[1]-bid_price)/par[2]) * Math.exp(-par[0]*(bid_price-par[1]) ) );

                    // derivative with respect to b
                    jacobian.setEntry(i, 1,     (Math.log(bid_price)*f*f*par[0]/par[2]) * Math.exp(-par[0]*(bid_price-par[1]) ) );

                    // derivative with respect to b
                    jacobian.setEntry(i, 2,     Math.log(bid_price)*f/par[2] );

                    //Some print statements so I can watch all of the internal calls to the fuction
                    System.out.println("Value of the model f: " + value.getEntry(i) );
                    System.out.println("Value of df/da      : " + jacobian.getEntry(i, 0) );
                    System.out.println("Value of df/db      : " + jacobian.getEntry(i, 1) );
                    System.out.println("Value of df/dc      : " + jacobian.getEntry(i, 2) );
                }
                System.out.println("Exit loop");
                System.out.println(" ");

                return new Pair<RealVector, RealMatrix>(value, jacobian);

            }
        };

        //Build the array of weights which will be the inverse of the uncertaintyy in the average win price.
        //Below is the inverse of the standard deviation on the mean (sqrt(n_wins)/(sigma)
        //Where sigma^2 = average(win_price^2) - average(win_price)*average(win_price)
        double[] weights = new double[bid_price_avg.length];
        for(int i=0; i<weights.length; i++) weights[i] = Math.sqrt( n_wins[i]/(win_price_avgsqr[i] -  win_price_avg[i]*win_price_avg[i]) );


        // least squares problem to solve : modeled radius should be close to target radius
        LeastSquaresProblem problem = new LeastSquaresBuilder().
                start(new double[]{0.2, 2, 3}).
                model(log_Logistic).
                target(win_price_avg).
                weight(new DiagonalMatrix(weights) ). //The weights must be a diagonal matrix.
                lazyEvaluation(false).
                maxEvaluations(1000).
                maxIterations(1000).
                build();
        LeastSquaresOptimizer.Optimum optimum = new LevenbergMarquardtOptimizer().optimize(problem);

        //These are the best fit parameters in order (a, b, c)
        double parList[] = new double[] {optimum.getPoint().getEntry(0),
                                         optimum.getPoint().getEntry(1),
                                         optimum.getPoint().getEntry(2)};

        //Print out the fit results.
        System.out.println("Best Fit Parameters: " + parList[0] + " " + parList[1] + " " + parList[2] );
        System.out.println("Chi2: " + optimum.getCost());
        System.out.println("evaluations: " + optimum.getEvaluations());
        System.out.println("iterations: " + optimum.getIterations());
    }
}
