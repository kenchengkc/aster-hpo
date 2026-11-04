#include "aster/accounting.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using Input = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Output = nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

NB_MODULE(_core, module) {
    module.attr("__version__") = ASTER_NATIVE_VERSION;
    module.def("build_info", [] {
        nb::dict result;
        result["compiler"] = __VERSION__;
        result["build_type"] = ASTER_BUILD_TYPE;
        result["flags"] = ASTER_BUILD_FLAGS;
        result["precision"] = "float64";
        result["fp_contract"] = "off";
        return result;
    });
    module.def("_hedge_pnl_into", [](Input prices, Input holdings,
               Output pnl, Output costs, Output turnover, double cost_rate) {
        const std::size_t paths = prices.shape(0);
        const std::size_t intervals = holdings.shape(1);
        if (holdings.shape(0) != paths || prices.shape(1) != intervals + 1)
            throw std::invalid_argument("Accounting shapes do not match");
        nb::gil_scoped_release release;
        aster::hedge_pnl({prices.data(), prices.size()}, {holdings.data(), holdings.size()},
                        paths, intervals, cost_rate, {pnl.data(), pnl.size()},
                        {costs.data(), costs.size()}, {turnover.data(), turnover.size()});
    }, nb::arg("prices").noconvert(), nb::arg("holdings").noconvert(),
       nb::arg("pnl").noconvert(), nb::arg("costs").noconvert(),
       nb::arg("turnover").noconvert(), nb::arg("cost_rate"));
    nb::class_<aster::Accounting>(module, "Accounting")
        .def_ro("pnl", &aster::Accounting::pnl)
        .def_ro("costs", &aster::Accounting::costs)
        .def_ro("turnover", &aster::Accounting::turnover);
    nb::class_<aster::Ledger>(module, "Ledger")
        .def(nb::init<double, double>(), nb::arg("initial_price"), nb::arg("cost_rate"))
        .def("update", &aster::Ledger::update, nb::arg("price"), nb::arg("target_holding"))
        .def("finish", &aster::Ledger::finish, nb::arg("terminal_price"))
        .def("reset", &aster::Ledger::reset, nb::arg("initial_price"), nb::arg("cost_rate"))
        .def("result", &aster::Ledger::result);
}
