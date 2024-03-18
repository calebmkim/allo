// Define a software emulation for ap_fixed
template <int W, int I>
class ap_fixed_emu {
public:
    // Constructor
    ap_fixed_emu() : value(0) {}

    // Constructor with value initialization
    ap_fixed_emu(double val) : value(val) {}

    // Conversion operator to double
    operator double() const {
        return value;
    }

    // Arithmetic operations (addition)
    ap_fixed_emu operator+(const ap_fixed_emu& other) const {
        return ap_fixed_emu(value + other.value);
    }

    // Arithmetic operations (subtraction)
    ap_fixed_emu operator-(const ap_fixed_emu& other) const {
        return ap_fixed_emu(value - other.value);
    }

    // Arithmetic operations (multiplication)
    ap_fixed_emu operator*(const ap_fixed_emu& other) const {
        return ap_fixed_emu(value * other.value);
    }

    // Arithmetic operations (division)
    ap_fixed_emu operator/(const ap_fixed_emu& other) const {
        return ap_fixed_emu(value / other.value);
    }

private:
    double value; // Use double for software emulation
};