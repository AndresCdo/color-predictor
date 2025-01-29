import * as React from "react";
import 'bootstrap/dist/css/bootstrap.min.css';

const Alert = React.forwardRef(({ className, variant = "primary", ...props }, ref) => (
  <div
    ref={ref}
    role="alert"
    className={`alert alert-${variant} ${className}`}
    {...props}
  />
));
Alert.displayName = "Alert";

const AlertTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h5
    ref={ref}
    className={`alert-heading ${className}`}
    {...props}
  />
));
AlertTitle.displayName = "AlertTitle";

const AlertDescription = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={`mb-0 ${className}`}
    {...props}
  />
));
AlertDescription.displayName = "AlertDescription";

export { Alert, AlertTitle, AlertDescription };
